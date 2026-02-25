"""Error handling utilities for the Portkey model provider."""

from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

# Error messages that indicate context window overflow.
CONTEXT_WINDOW_OVERFLOW_MESSAGES: frozenset[str] = frozenset(
    {
        "prompt is too long",
        "input is too long",
        "maximum context length",
        "context window",
        "context length exceeded",
        "too many tokens",
    }
)


def handle_api_error(error: Exception) -> None:
    """Map provider API errors to Strands exception types.

    Args:
        error: The original exception from the API call.

    Raises:
        ContextWindowOverflowException: If the error indicates context window overflow.
        ModelThrottledException: If the error indicates rate limiting or overload.
    """
    message = str(error).lower()

    if any(pattern in message for pattern in CONTEXT_WINDOW_OVERFLOW_MESSAGES):
        raise ContextWindowOverflowException(str(error)) from error

    if "rate" in message or "429" in message or "overloaded" in message or "529" in message:
        raise ModelThrottledException(str(error)) from error
