"""LLM 重试与候选切换分类：统一判断异常是否应该重试，以及重试耗尽后是否应该尝试下一个候选。

返回固定的归一化原因；返回 None 表示不应重试或切换到其他候选，
调用方会立即原样抛出异常。
"""

from collections.abc import Mapping
from typing import Any

from trans_novel.llm.errors import LLMError

# Normalized fallback reasons (AllModelsFailedError exposes only these categories).
EMPTY_RESPONSE = "empty_response"
RATE_LIMIT = "rate_limit"
TIMEOUT = "timeout"
CONNECTION = "connection"
HTTP_408 = "http_408"
HTTP_409 = "http_409"
SERVER_ERROR = "server_error"
MODEL_NOT_FOUND = "model_not_found"
PROVIDER_RETRY = "provider_retry"


class EmptyResponseError(LLMError):
    """A provider response did not contain usable message content."""


_RETRY_HEADER = "x-should-retry"

# 超时异常族：内置 TimeoutError（含 socket.timeout）、httpx/openai/urllib3 的
# 各类 Timeout*；判定先于 connection（ConnectTimeout 等同时属于连接族）。
_TIMEOUT_NAMES = frozenset(
    {
        "TimeoutError",
        "TimeoutException",
        "APITimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadTimeoutError",
        "ConnectTimeoutError",
    }
)

# 连接/网络异常族：DNS、proxy、remote/local protocol、connection-reset 归一化为 connection。
_CONNECTION_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionRefusedError",
        "ConnectionAbortedError",
        "RemoteDisconnected",
        "ProxyError",
        "ProtocolError",
        "NetworkError",
        "APIConnectionError",
        "ConnectError",
        "gaierror",
        "herror",
    }
)

# 匹配用小写化集合：候选类名剥离测试替身 `_` 前缀后小写对照
# （gaierror/herror 本身即小写）。
_TIMEOUT_NAMES_LC = frozenset(name.lower() for name in _TIMEOUT_NAMES)
_CONNECTION_NAMES_LC = frozenset(name.lower() for name in _CONNECTION_NAMES)


def _as_status(value: Any) -> int | None:
    """从整数或十进制字符串解析状态码；布尔/其它类型忽略。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value.strip()
        if digits.isdigit():
            return int(digits)
    return None


def _status_code(error: BaseException) -> int | None:
    """依次从异常 status_code、code、response.status_code 读取状态码。"""
    for attr in ("status_code", "code"):
        status = _as_status(getattr(error, attr, None))
        if status is not None:
            return status
    response = getattr(error, "response", None)
    if response is not None:
        status = _as_status(getattr(response, "status_code", None))
        if status is not None:
            return status
    return None


def _find_header(headers: Any, name: str) -> bool | None:
    """大小写不敏感地在 headers 容器里找布尔值；未识别值视为无该头部。"""
    try:
        items = headers.items()
    except AttributeError:
        return None
    for key, value in items:
        if key.lower() != name.lower():
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        return None
    return None


def _explicit_retry_header(error: BaseException) -> bool | None:
    """先 error.response.headers，再 error.headers。"""
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            header = _find_header(headers, _RETRY_HEADER)
            if header is not None:
                return header
    direct = getattr(error, "headers", None)
    if direct is not None:
        header = _find_header(direct, _RETRY_HEADER)
        if header is not None:
            return header
    return None


def _family_match(error: BaseException, names_lc: frozenset[str]) -> bool:
    """按规范化类名做精确匹配：沿 MRO 检查类名，剥离测试替身 `_` 前缀并小写后，
    精确命中固定族集合；绝不做子串判定——名称含 connect 的业务异常不会误判。
    """
    for cls in type(error).__mro__:
        name = cls.__name__.lstrip("_").lower()
        if name in names_lc:
            return True
    return False


def _is_timeout(error: BaseException) -> bool:
    # 可靠基类：内置 TimeoutError（py3 中 socket.timeout 即其别名）。
    if isinstance(error, TimeoutError):
        return True
    return _family_match(error, _TIMEOUT_NAMES_LC)


def _is_connection(error: BaseException) -> bool:
    # 可靠基类：内置 ConnectionError（含 Reset/Refused/Aborted 及
    # http.client.RemoteDisconnected）。
    if isinstance(error, ConnectionError):
        return True
    return _family_match(error, _CONNECTION_NAMES_LC)


def classify_retry(error: BaseException) -> str | None:
    """集中分类：返回可重试/可降级的固定 reason，或 None（永久失败）。

    优先级：显式 x-should-retry false > EmptyResponseError > 数字状态码
    > 超时/连接异常族 > 显式 x-should-retry true（provider_retry）。
    """
    header = _explicit_retry_header(error)
    if header is False:
        return None
    if isinstance(error, EmptyResponseError):
        return EMPTY_RESPONSE
    status = _status_code(error)
    if status is not None:
        if status == 408:
            return HTTP_408
        if status == 409:
            return HTTP_409
        if status == 429:
            return RATE_LIMIT
        if status >= 500:
            return SERVER_ERROR
        return None  # 其余 4xx/3xx 等永久
    if _is_timeout(error):
        return TIMEOUT
    if _is_connection(error):
        return CONNECTION
    if header is True:
        return PROVIDER_RETRY
    return None


def _structured_model_not_found(error: BaseException) -> bool:
    if getattr(error, "code", None) == MODEL_NOT_FOUND:
        return True
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return False
    if body.get("code") == MODEL_NOT_FOUND:
        return True
    nested = body.get("error")
    return isinstance(nested, Mapping) and nested.get("code") == MODEL_NOT_FOUND


def classify_fallback(error: BaseException) -> str | None:
    """Return a fixed reason only when an exhausted candidate may be skipped."""
    if _status_code(error) == 404 or _structured_model_not_found(error):
        return MODEL_NOT_FOUND
    return classify_retry(error)
