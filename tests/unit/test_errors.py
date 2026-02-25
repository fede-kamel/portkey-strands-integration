"""Unit tests for strands_portkey._errors."""

import pytest
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

from strands_portkey._errors import CONTEXT_WINDOW_OVERFLOW_MESSAGES, handle_api_error


class TestContextWindowOverflowMessages:
    def test_is_frozenset(self):
        assert isinstance(CONTEXT_WINDOW_OVERFLOW_MESSAGES, frozenset)

    def test_contains_expected_patterns(self):
        assert "prompt is too long" in CONTEXT_WINDOW_OVERFLOW_MESSAGES
        assert "maximum context length" in CONTEXT_WINDOW_OVERFLOW_MESSAGES
        assert "context window" in CONTEXT_WINDOW_OVERFLOW_MESSAGES
        assert "too many tokens" in CONTEXT_WINDOW_OVERFLOW_MESSAGES


class TestHandleApiError:
    def test_context_window_prompt_too_long(self):
        with pytest.raises(ContextWindowOverflowException):
            handle_api_error(Exception("prompt is too long: 200000 tokens"))

    def test_context_window_input_too_long(self):
        with pytest.raises(ContextWindowOverflowException):
            handle_api_error(Exception("input is too long for this model"))

    def test_context_window_maximum_context_length(self):
        with pytest.raises(ContextWindowOverflowException):
            handle_api_error(Exception("maximum context length exceeded for model gpt-4o"))

    def test_context_window_context_window(self):
        with pytest.raises(ContextWindowOverflowException):
            handle_api_error(Exception("context window exceeded"))

    def test_context_window_context_length_exceeded(self):
        with pytest.raises(ContextWindowOverflowException):
            handle_api_error(Exception("context length exceeded"))

    def test_context_window_too_many_tokens(self):
        with pytest.raises(ContextWindowOverflowException):
            handle_api_error(Exception("too many tokens in the request"))

    def test_throttled_rate_limit(self):
        with pytest.raises(ModelThrottledException):
            handle_api_error(Exception("rate limit exceeded"))

    def test_throttled_429(self):
        with pytest.raises(ModelThrottledException):
            handle_api_error(Exception("Error code: 429 - Too Many Requests"))

    def test_throttled_overloaded(self):
        with pytest.raises(ModelThrottledException):
            handle_api_error(Exception("Overloaded"))

    def test_throttled_529(self):
        with pytest.raises(ModelThrottledException):
            handle_api_error(Exception("529 overloaded"))

    def test_no_match_does_not_raise(self):
        # Should not raise anything for unrecognized errors
        handle_api_error(Exception("some random error"))

    def test_chained_exception_context_window(self):
        cause = ValueError("original")
        error = Exception("input is too long")
        error.__cause__ = cause
        with pytest.raises(ContextWindowOverflowException) as exc_info:
            handle_api_error(error)
        assert exc_info.value.__cause__ is error

    def test_chained_exception_throttled(self):
        error = Exception("rate limit exceeded")
        with pytest.raises(ModelThrottledException) as exc_info:
            handle_api_error(error)
        assert exc_info.value.__cause__ is error
