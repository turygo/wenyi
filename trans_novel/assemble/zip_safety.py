"""Bounded ZIP preflight and member reads shared by EPUB ingest/export."""

from __future__ import annotations

import lzma
import stat
import zipfile
import zlib

MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10000


class ZipSafetyError(ValueError):
    """A deterministic, sanitized archive safety failure."""

    def __init__(self, code: str, name: str = "") -> None:
        self.code = code
        self.name = name
        super().__init__(code)


def canonical_name(name: str) -> str | None:
    """Return the member identity while retaining a legal directory slash."""
    if not safe_name(name):
        return None
    return name[:-1] if name.endswith("/") else name


def safe_name(name: str) -> bool:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return False
    parts = name.split("/")
    # A single trailing slash denotes a directory. Empty interior segments
    # and dot segments are aliases rather than legal member names.
    if any(part in {"", ".", ".."} for part in parts[:-1]):
        return False
    return parts[-1] not in {".", ".."} and (parts[-1] != "" or name.endswith("/"))


def regular_member(info: zipfile.ZipInfo) -> bool:
    """Allow regular files and directories, reject symlinks/special files."""
    mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(mode)
    if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        return False
    return not stat.S_ISLNK(mode)


def read_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_member_bytes: int = MAX_MEMBER_BYTES,
) -> bytes:
    if info.file_size > max_member_bytes:
        raise ZipSafetyError("member_too_large", info.filename)
    chunks: list[bytes] = []
    total = 0
    try:
        with zf.open(info, "r") as stream:
            while True:
                chunk = stream.read(min(1024 * 1024, max_member_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_member_bytes:
                    raise ZipSafetyError("member_too_large", info.filename)
                chunks.append(chunk)
    except ZipSafetyError:
        raise
    except (
        OSError,
        RuntimeError,
        NotImplementedError,
        EOFError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ValueError,
        zlib.error,
        lzma.LZMAError,
    ) as exc:
        code = "crc_error" if type(exc).__name__ in {"BadZipFile", "BadCRC"} else "member_read"
        raise ZipSafetyError(code, info.filename) from None
    if total != info.file_size:
        raise ZipSafetyError("member_size_mismatch", info.filename)
    return b"".join(chunks)


def preflight_zip(
    zf: zipfile.ZipFile,
    *,
    read_members: bool = True,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_archive_bytes: int = MAX_ARCHIVE_BYTES,
    max_archive_members: int = MAX_ARCHIVE_MEMBERS,
) -> list[zipfile.ZipInfo]:
    """Validate names/types/count/declared and actual bounded bytes before parsing."""
    try:
        infos = zf.infolist()
    except (OSError, zipfile.BadZipFile):
        raise ZipSafetyError("invalid_zip") from None
    if len(infos) > max_archive_members:
        raise ZipSafetyError("archive_member_limit")
    seen: set[str] = set()
    declared = 0
    for info in infos:
        identity = canonical_name(info.filename)
        if identity is None:
            raise ZipSafetyError("unsafe_entry", info.filename)
        if identity in seen:
            raise ZipSafetyError("duplicate_entry", info.filename)
        seen.add(identity)
        if info.flag_bits & 0x1:
            raise ZipSafetyError("encrypted_entry", info.filename)
        if not regular_member(info):
            raise ZipSafetyError("special_entry", info.filename)
        if info.file_size > max_member_bytes:
            raise ZipSafetyError("member_too_large", info.filename)
        declared += info.file_size
        if declared > max_archive_bytes:
            raise ZipSafetyError("archive_too_large")
    if read_members:
        for info in infos:
            read_member(zf, info, max_member_bytes=max_member_bytes)
    return infos


__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_ARCHIVE_MEMBERS",
    "MAX_MEMBER_BYTES",
    "ZipSafetyError",
    "canonical_name",
    "preflight_zip",
    "read_member",
    "regular_member",
    "safe_name",
]
