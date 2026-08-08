"""Test suite for the LLM retry utility.

Covers `_is_tool_use_failed` (happy path, a different error code, and a
missing/malformed error body handled defensively) and `invoke_with_retry`
(first-attempt success, a mid-loop success after one retriable failure,
every attempt exhausted, and a non-retriable error raised immediately
without consuming a retry).

"""

from unittest.mock import MagicMock

import httpx
import pytest
from groq import BadRequestError

from agent.utils.llm_utils import _is_tool_use_failed, invoke_with_retry


def _make_bad_request_error(body: dict | None) -> BadRequestError:
    """Build a real BadRequestError with the given error body, for testing."""
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.groq.com/test"),
    )
    return BadRequestError("test message", response=response, body=body)


def test_is_tool_use_failed_true_for_matching_code() -> None:
    """Verify a tool_use_failed error code is correctly identified."""
    exc = _make_bad_request_error({"error": {"code": "tool_use_failed"}})
    assert _is_tool_use_failed(exc) is True


def test_is_tool_use_failed_false_for_other_code() -> None:
    """Verify a different error code is not treated as tool_use_failed."""
    exc = _make_bad_request_error({"error": {"code": "invalid_api_key"}})
    assert _is_tool_use_failed(exc) is False


def test_is_tool_use_failed_false_for_missing_body() -> None:
    """Verify a missing/None error body is handled defensively, not raised."""
    exc = _make_bad_request_error(None)
    assert _is_tool_use_failed(exc) is False


def test_is_tool_use_failed_false_for_body_without_error_key() -> None:
    """Verify a body missing the 'error' key is handled defensively."""
    exc = _make_bad_request_error({"something_else": "unexpected"})
    assert _is_tool_use_failed(exc) is False


def test_invoke_with_retry_succeeds_on_first_attempt() -> None:
    """Verify a successful first call returns immediately, no retry needed."""
    mock_llm = MagicMock()
    mock_result = MagicMock()
    mock_llm.invoke.return_value = mock_result

    result = invoke_with_retry(mock_llm, messages=[])

    assert result is mock_result
    assert mock_llm.invoke.call_count == 1


def test_invoke_with_retry_succeeds_after_one_retriable_failure() -> None:
    """Verify a tool_use_failed error is retried and can succeed on a later attempt."""
    mock_llm = MagicMock()
    mock_result = MagicMock()
    mock_llm.invoke.side_effect = [
        _make_bad_request_error({"error": {"code": "tool_use_failed"}}),
        mock_result,
    ]

    result = invoke_with_retry(mock_llm, messages=[], max_retries=3)

    assert result is mock_result
    assert mock_llm.invoke.call_count == 2


def test_invoke_with_retry_raises_after_exhausting_all_attempts() -> None:
    """Verify the last exception is raised once every retry is exhausted."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = _make_bad_request_error(
        {"error": {"code": "tool_use_failed"}}
    )

    with pytest.raises(BadRequestError) as exc_info:
        invoke_with_retry(mock_llm, messages=[], max_retries=3)

    assert exc_info.value.body["error"]["code"] == "tool_use_failed"
    assert mock_llm.invoke.call_count == 3


def test_invoke_with_retry_raises_immediately_for_non_retriable_error() -> None:
    """Verify a non-tool_use_failed error is raised right away, no retry consumed."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = _make_bad_request_error(
        {"error": {"code": "invalid_api_key"}}
    )

    with pytest.raises(BadRequestError):
        invoke_with_retry(mock_llm, messages=[], max_retries=3)

    assert mock_llm.invoke.call_count == 1
