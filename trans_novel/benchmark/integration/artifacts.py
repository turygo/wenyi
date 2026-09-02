# fmt: off

"""Canonical integration artifact publication and validation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import (
    ArtifactError,
    artifact_sha256,
    atomic_json,
    read_canonical_json,
    read_json,
    relative_path,
)
from trans_novel.benchmark.run import BenchmarkError, validate_candidate_capabilities
from trans_novel.benchmark.schema import CandidateSpec
from trans_novel.llm.telemetry import CallAttemptTelemetry

_HEX64 = '^[0-9a-f]{64}$'
_SAFE_ID = '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'

class IntegrationError(BenchmarkError):
    """Invalid immutable input, state, or integration evidence."""

class IntegrationIntegrityError(IntegrationError):
    """Corrupt persisted lineage that must abort the whole integration."""

def integration_sha256(path: Path) -> str:
    try:
        return artifact_sha256(path)
    except (ArtifactError, OSError) as error:
        raise IntegrationError(f'cannot hash {path}: {error}') from error

def integration_relative_path(path: Path, root: Path) -> str:
    try:
        return relative_path(path, root)
    except ArtifactError as error:
        raise IntegrationError(str(error)) from error

def read_integration_json(path: Path) -> Any:
    return read_json(path, error_type=IntegrationError)

def read_integration_canonical(path: Path) -> Any:
    return read_canonical_json(path, error_type=IntegrationError)

def write_integration_json(path: Path, value: Any) -> None:
    atomic_json(path, value)

def read_telemetry_records(path: Path) -> list[CallAttemptTelemetry]:
    try:
        return [CallAttemptTelemetry.model_validate(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception as error:
        raise IntegrationError(f'invalid telemetry artifact {path}: {error}') from error

def translator_call_count(records: list[CallAttemptTelemetry]) -> int:
    return sum(r.operation in {'translate.batch', 'translate.single', 'translate.heading'} and r.attempt_index == 1 for r in records)

def usage_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {'unknown_required_usage_count': 1, 'valid': False}
    try:
        value = read_integration_json(path)
        if value.get('schema_version') != 2:
            raise ValueError('usage schema mismatch')
        totals = value['totals']
        required = ('calls', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cache_hit_tokens', 'cache_miss_tokens')
        if any(isinstance(totals.get(k), bool) or not isinstance(totals.get(k), int) or totals[k] < 0 for k in required):
            raise ValueError('invalid usage totals')
    except Exception:
        return {'unknown_required_usage_count': 1, 'valid': False}
    slots = [v for v in (value.get('by_agent') or {}).values() if isinstance(v, dict)]
    return {'unknown_required_usage_count': 0, 'valid': True, 'calls': int(totals['calls']), 'attempts': sum(int(v.get('attempts', 0)) for v in slots), 'logical_calls': sum(int(v.get('logical_calls', 0)) for v in slots), **{k: int(totals[k]) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}

def validate_candidate_paths(root: Path, source: Path, *paths: Path) -> None:
    resolved_root, resolved_source = (root.resolve(), source.resolve())
    resolved = []
    for path in paths:
        current = path.resolve(strict=False)
        try:
            current.relative_to(resolved_root)
        except ValueError as error:
            raise IntegrationError(f'candidate path escapes integration root: {path}') from error
        resolved.append(current)
    values = [str(p).casefold() for p in (resolved_source, *resolved)]
    if len(values) != len(set(values)):
        raise IntegrationError('candidate artifact paths alias')

def validate_all_candidate_paths(root: Path, source: Path, candidate_ids: list[str], book_id: str) -> None:
    planned = []
    for cid in candidate_ids:
        candidate = root / 'candidates' / cid
        state, output = (candidate / 'state', candidate / 'outputs')
        planned.extend((candidate, state, output, output / f'{book_id}.epub', output / f'{book_id}-bi.epub', candidate / 'telemetry.jsonl', candidate / 'telemetry.first.jsonl', candidate / 'telemetry.resume.jsonl', candidate / 'canary.json', candidate / 'result.json'))
    validate_candidate_paths(root, source, *planned)

def _validate_request(request: dict[str, Any]) -> None:
    required = ('schema_version', 'benchmark_id', 'corpus_sha256', 'book_spec_sha256', 'candidate_spec_sha256', 'integration_spec_sha256', 'book_id', 'source_sha256', 'source_language', 'target_language', 'candidate_ids', 'candidates', 'interrupt_after_committed_batches', 'output_mono', 'output_bilingual', 'bilingual_order')
    if any(k not in request for k in required):
        raise IntegrationError('integration request contract is incomplete')
    ids = request['candidate_ids']
    if request['schema_version'] != 1 or not isinstance(request['benchmark_id'], str) or (not isinstance(request['book_id'], str)) or (not isinstance(ids, list)) or (len(ids) not in {2, 3}) or any(not isinstance(cid, str) or not re.fullmatch(_SAFE_ID, cid) for cid in ids) or (len({cid.casefold() for cid in ids}) != len(ids)) or (not isinstance(request['candidates'], dict)) or (set(request['candidates']) != set(ids)) or (request['source_language'] != 'en') or (request['target_language'] != 'zh') or (request['output_mono'] is not True) or (request['output_bilingual'] is not True) or (request['bilingual_order'] not in {'target_first', 'source_first'}) or (type(request['interrupt_after_committed_batches']) is not int) or (request['interrupt_after_committed_batches'] < 1) or any(not isinstance(request.get(k), str) or not re.fullmatch(_HEX64, request[k]) for k in ('corpus_sha256', 'book_spec_sha256', 'candidate_spec_sha256', 'integration_spec_sha256', 'source_sha256')):
        raise IntegrationError('integration request contract is invalid')

def _validate_request_candidates(request: dict[str, Any]) -> None:
    fields = ('pipeline_variant', 'translator_model', 'analyst_model', 'editor_model', 'fast_model', 'temperature', 'seed')
    for cid, candidate in request['candidates'].items():
        if not isinstance(candidate, dict) or any(k not in candidate for k in fields) or (not isinstance(candidate.get('pipeline_variant'), str)) or (candidate.get('pipeline_variant') not in {'minimal', 'polish'}) or any(not isinstance(candidate.get(k), str) for k in ('translator_model', 'analyst_model', 'editor_model', 'fast_model')) or isinstance(candidate.get('temperature'), bool) or (not isinstance(candidate.get('temperature'), int | float)) or (candidate.get('seed') is not None and (isinstance(candidate.get('seed'), bool) or not isinstance(candidate.get('seed'), int))):
            raise IntegrationError(f'integration request candidate {cid} is invalid')
    if any(c['temperature'] != 0.1 or c['seed'] is not None for c in request['candidates'].values()):
        raise IntegrationError('integration candidates require fixed generation controls')
    try:
        spec = CandidateSpec.model_validate({'schema_version': 3, 'benchmark_id': request['benchmark_id'], 'temperature': 0.1, 'seed': None, 'replicates': 1, 'candidates': [{'candidate_id': cid, **{k: request['candidates'][cid][k] for k in ('translator_model', 'analyst_model', 'editor_model', 'fast_model', 'pipeline_variant')}} for cid in request['candidate_ids']]})
        validate_candidate_capabilities(spec)
    except Exception as error:
        raise IntegrationError(f'integration candidate capabilities are invalid: {error}') from error

def _evidence_path(relative: Any, out: Path, message: str) -> Path:
    if not isinstance(relative, str):
        raise IntegrationIntegrityError(message)
    path = (out / relative).resolve()
    try:
        valid = integration_relative_path(path, out) == relative
    except IntegrationError as error:
        raise IntegrationIntegrityError(message) from error
    if not valid:
        raise IntegrationIntegrityError(message)
    return path

def validate_terminal_artifacts(root: Path | str) -> dict[str, Any]:
    out = Path(root).expanduser().resolve()
    request_path, integration_path, complete_path = (out / name for name in ('integration_request.json', 'integration.json', 'integration_complete.json'))
    request, integration, complete = (read_integration_canonical(request_path), read_integration_canonical(integration_path), read_integration_canonical(complete_path))
    if not all(isinstance(v, dict) for v in (request, integration, complete)):
        raise IntegrationError('terminal integration artifacts must be objects')
    _validate_request(request)
    _validate_request_candidates(request)
    lineage = ('schema_version', 'benchmark_id', 'corpus_sha256', 'book_spec_sha256', 'candidate_spec_sha256', 'integration_spec_sha256', 'book_id', 'source_sha256')
    if any(integration.get(k) != request[k] for k in lineage):
        raise IntegrationError('integration manifest request lineage mismatch')
    if not isinstance(integration.get('candidates'), dict) or not isinstance(complete.get('candidates'), dict) or complete.get('schema_version') != 1 or (complete.get('benchmark_id') != request['benchmark_id']) or (complete.get('integration_sha256') != integration_sha256(integration_path)) or (complete.get('terminal') is not True) or (set(integration['candidates']) != set(request['candidate_ids'])) or (set(complete['candidates']) != set(request['candidate_ids'])):
        raise IntegrationError('integration completion manifest is invalid')
    candidates, request_hash = ({}, integration_sha256(request_path))
    for cid in sorted(request['candidate_ids']):
        entry, centry = (integration['candidates'].get(cid), complete['candidates'].get(cid))
        if not isinstance(entry, dict) or not isinstance(centry, dict) or (not isinstance(entry.get('result_path'), str)) or Path(entry['result_path']).is_absolute() or ('..' in Path(entry['result_path']).parts) or (centry.get('result_path') != entry['result_path']) or (centry.get('result_sha256') != entry.get('result_sha256')) or (centry.get('status') not in {'completed', 'failed'}) or (not isinstance(entry.get('result_sha256'), str)) or (not re.fullmatch(_HEX64, entry['result_sha256'])):
            raise IntegrationError(f'integration candidate {cid} manifest is invalid')
        path = (out / entry['result_path']).resolve()
        if integration_relative_path(path, out) != entry['result_path'] or not path.is_file():
            raise IntegrationError(f'integration candidate {cid} result path is invalid')
        if integration_sha256(path) != entry['result_sha256']:
            raise IntegrationIntegrityError(f'integration candidate {cid} result hash mismatch')
        result = read_integration_canonical(path)
        if not isinstance(result, dict):
            raise IntegrationError(f'integration candidate {cid} result is invalid')
        validate_result_contract(result, candidate_id=cid, expected_lineage={k: request[k] for k in lineage}, request_sha256=request_hash, status=centry['status'], out=out)
        if (centry['status'] == 'completed') != result['passed']:
            raise IntegrationError(f'integration candidate {cid} status mismatch')
        candidates[cid] = {'result': result, 'result_path': entry['result_path'], 'result_sha256': entry['result_sha256'], 'status': centry['status']}
    return {'request': request, 'request_sha256': request_hash, 'integration': integration, 'integration_sha256': integration_sha256(integration_path), 'complete': complete, 'integration_complete_sha256': integration_sha256(complete_path), 'candidates': candidates, 'root': out}
def _validate_present_failed_evidence(value: dict[str, Any], out: Path) -> None:
    if 'output_paths' in value or 'output_sha256' in value:
        paths, hashes = value.get('output_paths'), value.get('output_sha256')
        if not isinstance(paths, dict) or not isinstance(hashes, dict):
            raise IntegrationIntegrityError('integration output evidence is invalid')
        for name in set(paths) | set(hashes):
            relative, digest = paths.get(name), hashes.get(name)
            path = _evidence_path(relative, out, 'integration output evidence is missing or tampered')
            if not isinstance(digest, str) or not re.fullmatch(_HEX64, digest) or not path.is_file() or integration_sha256(path) != digest:
                raise IntegrationIntegrityError('integration output evidence is missing or tampered')
    physical = {}
    for path_key, hash_key in (('telemetry_path', 'telemetry_sha256'), ('first_telemetry_path', 'first_telemetry_sha256'), ('resume_telemetry_path', 'resume_telemetry_sha256'), ('usage_path', 'usage_sha256')):
        if path_key not in value and hash_key not in value:
            continue
        relative, digest = value.get(path_key), value.get(hash_key)
        path = _evidence_path(relative, out, 'integration physical evidence is missing or tampered')
        if not isinstance(digest, str) or not re.fullmatch(_HEX64, digest) or not path.is_file() or integration_sha256(path) != digest:
            raise IntegrationIntegrityError('integration physical evidence is missing or tampered')
        physical[path_key] = path
    records = {k: read_telemetry_records(p) for k, p in physical.items() if k.endswith('telemetry_path')}
    for path_key, count_key in (('first_telemetry_path', 'first_telemetry_count'), ('resume_telemetry_path', 'resume_telemetry_count'), ('resume_telemetry_path', 'resume_attempt_telemetry_count')):
        if path_key in records and count_key in value and (type(value[count_key]) is not int or value[count_key] < 0 or value[count_key] != len(records[path_key])):
            raise IntegrationIntegrityError('integration telemetry record count mismatch')
    if 'telemetry_path' in records and isinstance(value.get('telemetry_counts'), dict):
        rows, counts = records['telemetry_path'], value['telemetry_counts']
        expected = {'logical_call_count': len({r.logical_call_id for r in rows}), 'attempt_count': len(rows), 'operation_count': len({r.operation for r in rows}), 'agent_count': len({r.agent for r in rows}), 'retry_count': sum(r.attempt_index > 1 or r.retry_class is not None for r in rows), 'translate_call_count': translator_call_count(rows)}
        if any(counts.get(k) != v for k, v in expected.items()):
            raise IntegrationIntegrityError('integration telemetry count mismatch')
    if 'usage_path' in physical and not usage_evidence(physical['usage_path']).get('valid'):
        raise IntegrationIntegrityError('integration usage evidence is invalid')
    for key in ('before_target_hashes', 'final_target_hashes'):
        if key in value and (not isinstance(value[key], list) or any(not isinstance(r, dict) or type(r.get('chapter')) is not int or r['chapter'] < 0 or type(r.get('start')) is not int or r['start'] < 0 or type(r.get('count')) is not int or r['count'] < 0 or not isinstance(r.get('target_sha256'), str) or not re.fullmatch(_HEX64, r['target_sha256']) for r in value[key])):
            raise IntegrationIntegrityError('integration slice evidence is invalid')

def validate_result_contract(value: dict[str, Any], *, candidate_id: str, expected_lineage: dict[str, Any], request_sha256: str, status: str, out: Path) -> None:
    if value.get('schema_version') != 1 or value.get('candidate_id') != candidate_id:
        raise IntegrationError('integration result schema mismatch')
    lineage = ('corpus_sha256', 'book_spec_sha256', 'candidate_spec_sha256', 'integration_spec_sha256', 'source_sha256', 'benchmark_id', 'book_id')
    if any(value.get(k) != expected_lineage.get(k) for k in lineage):
        raise IntegrationError('integration result lineage mismatch')
    if value.get('request_sha256') != request_sha256 or type(value.get('passed')) is not bool:
        raise IntegrationError('integration result request lineage mismatch')
    base = ('resume_duplicate_operations', 'reasoning_tokens', 'model_mismatch_count', 'unknown_required_usage_count')
    if any(type(value.get(k)) is not int or value[k] < 0 for k in base):
        raise IntegrationError('integration result counters are invalid')
    if any(type(value.get(k)) is not bool for k in ('canary_passed', 'expected_interruption_observed', 'readiness_passed')):
        raise IntegrationError('integration result predicates are invalid')
    if any(not isinstance(value.get(k), dict) or type(value[k].get('structural_pass')) is not bool for k in ('structural', 'mono', 'bilingual')):
        raise IntegrationError('integration structural predicates are invalid')
    if status == 'failed':
        if value['passed']:
            raise IntegrationError('failed integration result pass predicate is invalid')
        _validate_present_failed_evidence(value, out)
        return
    if status != 'completed':
        raise IntegrationError('integration result status is invalid')
    counters = (*base, 'committed_batches', 'skipped_batches', 'remaining_batches', 'repeated_batches')
    if any(type(value.get(k)) is not int or value[k] < 0 for k in counters):
        raise IntegrationError('completed integration result counters are invalid')
    evidence = bool(value['canary_passed'] and value['expected_interruption_observed'] and value['readiness_passed'] and (value['resume_duplicate_operations'] == 0) and value['structural']['structural_pass'] and value['mono']['structural_pass'] and value['bilingual']['structural_pass'] and (value['reasoning_tokens'] == 0) and (value['model_mismatch_count'] == 0) and (value['unknown_required_usage_count'] == 0))
    if value['passed'] != evidence:
        raise IntegrationError('integration result pass evidence mismatch')
    if not evidence:
        return
    timings, paths, hashes = (value.get('phase_timings_ms'), value.get('output_paths'), value.get('output_sha256'))
    if not isinstance(timings, dict) or any(type(timings.get(k)) is not int or timings[k] < 0 for k in ('prepare', 'translate', 'quality')) or any(type(timings.get(k)) is not int or timings[k] <= 0 for k in ('first_attempt', 'resume', 'total')) or (timings['total'] < timings['first_attempt'] + timings['resume']):
        raise IntegrationError('integration phase timing evidence is missing')
    if not isinstance(paths, dict) or not isinstance(hashes, dict) or set(paths) != {'mono', 'bilingual'} or (set(hashes) != {'mono', 'bilingual'}):
        raise IntegrationError('integration output evidence is missing')
    for key in paths:
        path = _evidence_path(paths[key], out, 'integration output evidence is missing or tampered')
        if not path.is_file() or integration_sha256(path) != hashes.get(key):
            raise IntegrationIntegrityError('integration output evidence is missing or tampered')
    physical = {}
    for key, digest_key in (('telemetry_path', 'telemetry_sha256'), ('first_telemetry_path', 'first_telemetry_sha256'), ('resume_telemetry_path', 'resume_telemetry_sha256'), ('usage_path', 'usage_sha256')):
        relative, digest = value.get(key), value.get(digest_key)
        path = _evidence_path(relative, out, 'integration physical evidence is missing or tampered')
        if not isinstance(digest, str) or not path.is_file() or integration_sha256(path) != digest:
            raise IntegrationIntegrityError('integration physical evidence is missing or tampered')
        physical[key] = path
    telemetry = read_telemetry_records(physical['telemetry_path'])
    first, resume = (read_telemetry_records(physical['first_telemetry_path']), read_telemetry_records(physical['resume_telemetry_path']))
    for key, rows in (('first_telemetry_count', first), ('resume_telemetry_count', resume)):
        if type(value.get(key)) is not int or value[key] < 0 or value[key] != len(rows):
            raise IntegrationIntegrityError('integration telemetry record count mismatch')
    attempt = value.get('resume_attempt_telemetry_count')
    if type(attempt) is not int or attempt < 0 or attempt > len(resume):
        raise IntegrationIntegrityError('integration telemetry record count mismatch')
    counts = value.get('telemetry_counts')
    expected = {'logical_call_count': len({r.logical_call_id for r in telemetry}), 'attempt_count': len(telemetry), 'operation_count': len({r.operation for r in telemetry}), 'agent_count': len({r.agent for r in telemetry}), 'retry_count': sum(r.attempt_index > 1 or r.retry_class is not None for r in telemetry), 'translate_call_count': translator_call_count(telemetry)}
    if not isinstance(counts, dict) or any((type(counts.get(k)) is not int or counts[k] < 0 or counts[k] != v for k, v in expected.items())):
        raise IntegrationIntegrityError('integration telemetry count mismatch')
    usage = usage_evidence(physical['usage_path'])
    if not usage.get('valid') or usage.get('unknown_required_usage_count') != 0:
        raise IntegrationIntegrityError('integration usage evidence is invalid')
__all__ = ['IntegrationError', 'IntegrationIntegrityError', 'integration_relative_path', 'integration_sha256', 'read_integration_canonical', 'read_integration_json', 'read_telemetry_records', 'translator_call_count', 'usage_evidence', 'validate_all_candidate_paths', 'validate_candidate_paths', 'validate_result_contract', 'validate_terminal_artifacts', 'write_integration_json']

# fmt: on
