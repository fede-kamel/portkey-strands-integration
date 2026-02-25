"""Unit tests for strands_portkey._formatting."""

import json
from unittest.mock import MagicMock

import pytest

from strands_portkey._formatting import (
    format_chunk,
    format_request_message_content,
    format_request_message_tool_call,
    format_request_messages,
    format_request_tool_message,
    format_tool_choice,
)

# =============================================================================
# format_request_message_content
# =============================================================================


class TestFormatRequestMessageContent:
    def test_text_content(self):
        result = format_request_message_content({"text": "Hello world"})
        assert result == {"text": "Hello world", "type": "text"}

    def test_image_content_png(self):
        result = format_request_message_content(
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": b"fake-image-data"},
                }
            }
        )
        assert result["type"] == "image_url"
        assert "image_url" in result
        assert result["image_url"]["detail"] == "auto"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_content_jpeg(self):
        result = format_request_message_content({"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8"}}})
        assert "image/jpeg" in result["image_url"]["url"]

    def test_unsupported_content_raises(self):
        with pytest.raises(TypeError, match="unsupported type"):
            format_request_message_content({"audio": "something"})


# =============================================================================
# format_request_message_tool_call
# =============================================================================


class TestFormatRequestMessageToolCall:
    def test_basic_tool_call(self):
        tool_use = {
            "toolUseId": "tool-abc",
            "name": "calculator",
            "input": {"expression": "2+2"},
        }
        result = format_request_message_tool_call(tool_use)
        assert result["id"] == "tool-abc"
        assert result["type"] == "function"
        assert result["function"]["name"] == "calculator"
        assert json.loads(result["function"]["arguments"]) == {"expression": "2+2"}

    def test_empty_input(self):
        tool_use = {"toolUseId": "t1", "name": "noop", "input": {}}
        result = format_request_message_tool_call(tool_use)
        assert json.loads(result["function"]["arguments"]) == {}


# =============================================================================
# format_request_tool_message
# =============================================================================


class TestFormatRequestToolMessage:
    def test_text_tool_result(self):
        tool_result = {
            "toolUseId": "tool-abc",
            "content": [{"text": "The answer is 4"}],
        }
        result = format_request_tool_message(tool_result)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "tool-abc"
        assert result["content"][0]["text"] == "The answer is 4"

    def test_json_tool_result(self):
        tool_result = {
            "toolUseId": "tool-abc",
            "content": [{"json": {"result": 42}}],
        }
        result = format_request_tool_message(tool_result)
        assert result["role"] == "tool"
        content_text = result["content"][0]["text"]
        assert json.loads(content_text) == {"result": 42}


# =============================================================================
# format_request_messages
# =============================================================================


class TestFormatRequestMessages:
    def test_simple_user_message(self):
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        result = format_request_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_with_system_prompt(self):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = format_request_messages(messages, system_prompt="You are a helper")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helper"
        assert result[1]["role"] == "user"

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "user", "content": [{"text": "What is 2+2?"}]},
            {"role": "assistant", "content": [{"text": "4"}]},
            {"role": "user", "content": [{"text": "And 3+3?"}]},
        ]
        result = format_request_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_message_with_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "Let me calculate that."},
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "calc",
                            "input": {"x": 1},
                        }
                    },
                ],
            }
        ]
        result = format_request_messages(messages)
        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"
        assert result[0]["content"][0]["text"] == "Let me calculate that."

    def test_message_with_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"text": "result: 42"}],
                        }
                    }
                ],
            }
        ]
        result = format_request_messages(messages)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "t1"

    def test_filters_empty_messages(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "calc",
                            "input": {},
                        }
                    }
                ],
            }
        ]
        result = format_request_messages(messages)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "tool_calls" in assistant_msgs[0]

    def test_messages_with_mixed_content(self):
        """Messages with text, tool use, and tool result together."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"text": "I'll help."},
                    {"toolUse": {"toolUseId": "t1", "name": "search", "input": {"q": "test"}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "t1", "content": [{"text": "found it"}]}},
                    {"text": "Thanks, now summarize."},
                ],
            },
        ]
        result = format_request_messages(messages)
        roles = [m["role"] for m in result]
        assert "assistant" in roles
        assert "tool" in roles
        assert "user" in roles


# =============================================================================
# format_request_messages — system_prompt_content
# =============================================================================


class TestSystemPromptContent:
    def test_single_block(self):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = format_request_messages(messages, system_prompt_content=[{"text": "You are helpful."}])
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "You are helpful."

    def test_multiple_blocks_concatenated(self):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = format_request_messages(
            messages,
            system_prompt_content=[
                {"text": "You are helpful."},
                {"text": "Be concise."},
            ],
        )
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert "You are helpful." in sys_msgs[0]["content"]
        assert "Be concise." in sys_msgs[0]["content"]

    def test_cache_point_blocks_are_ignored(self):
        """Cache point blocks are ignored, text is extracted."""
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = format_request_messages(
            messages,
            system_prompt_content=[
                {"text": "Important context."},
                {"cachePoint": {"type": "default"}},
                {"text": "More context."},
            ],
        )
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert "Important context." in sys_msgs[0]["content"]
        assert "More context." in sys_msgs[0]["content"]

    def test_overrides_system_prompt_string(self):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = format_request_messages(
            messages,
            system_prompt="This should be ignored",
            system_prompt_content=[{"text": "This takes precedence."}],
        )
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "This takes precedence."

    def test_empty_list_produces_no_system_message(self):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = format_request_messages(messages, system_prompt_content=[])
        sys_msgs = [m for m in result if m["role"] == "system"]
        assert len(sys_msgs) == 0


# =============================================================================
# format_tool_choice
# =============================================================================


class TestToolChoice:
    def test_auto(self):
        assert format_tool_choice({"auto": {}}) == "auto"

    def test_any_maps_to_required(self):
        assert format_tool_choice({"any": {}}) == "required"

    def test_specific_tool(self):
        result = format_tool_choice({"tool": {"name": "calculator"}})
        assert result == {"type": "function", "function": {"name": "calculator"}}

    def test_unknown_key_defaults_to_auto(self):
        """Fallback branch: unrecognized dict key returns 'auto'."""
        result = format_tool_choice({})
        assert result == "auto"


# =============================================================================
# format_chunk
# =============================================================================


class TestFormatChunk:
    def test_message_start(self):
        assert format_chunk({"chunk_type": "message_start"}) == {"messageStart": {"role": "assistant"}}

    def test_content_start_text(self):
        assert format_chunk({"chunk_type": "content_start", "data_type": "text"}) == {
            "contentBlockStart": {"start": {}}
        }

    def test_content_start_tool(self):
        data = MagicMock()
        data.function.name = "calculator"
        data.id = "tool-123"
        result = format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": data})
        assert result["contentBlockStart"]["start"]["toolUse"]["name"] == "calculator"
        assert result["contentBlockStart"]["start"]["toolUse"]["toolUseId"] == "tool-123"

    def test_content_delta_text(self):
        result = format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": "Hello"})
        assert result == {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    def test_content_delta_tool(self):
        data = MagicMock()
        data.function.arguments = '{"x": 1}'
        result = format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": data})
        assert result["contentBlockDelta"]["delta"]["toolUse"]["input"] == '{"x": 1}'

    def test_content_delta_tool_empty_args(self):
        data = MagicMock()
        data.function.arguments = ""
        result = format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": data})
        assert result["contentBlockDelta"]["delta"]["toolUse"]["input"] == ""

    def test_content_delta_tool_none_args(self):
        data = MagicMock()
        data.function.arguments = None
        result = format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": data})
        assert result["contentBlockDelta"]["delta"]["toolUse"]["input"] == ""

    def test_content_delta_reasoning(self):
        result = format_chunk({"chunk_type": "content_delta", "data_type": "reasoning_content", "data": "thinking..."})
        assert result["contentBlockDelta"]["delta"]["reasoningContent"]["text"] == "thinking..."

    def test_content_stop(self):
        assert format_chunk({"chunk_type": "content_stop"}) == {"contentBlockStop": {}}

    def test_message_stop_end_turn(self):
        assert format_chunk({"chunk_type": "message_stop", "data": "stop"}) == {
            "messageStop": {"stopReason": "end_turn"}
        }

    def test_message_stop_tool_calls(self):
        assert format_chunk({"chunk_type": "message_stop", "data": "tool_calls"}) == {
            "messageStop": {"stopReason": "tool_use"}
        }

    def test_message_stop_length(self):
        assert format_chunk({"chunk_type": "message_stop", "data": "length"}) == {
            "messageStop": {"stopReason": "max_tokens"}
        }

    def test_metadata(self):
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 20
        usage.total_tokens = 30
        result = format_chunk({"chunk_type": "metadata", "data": usage})
        assert result["metadata"]["usage"]["inputTokens"] == 10
        assert result["metadata"]["usage"]["outputTokens"] == 20
        assert result["metadata"]["usage"]["totalTokens"] == 30

    def test_unknown_chunk_type_raises(self):
        with pytest.raises(RuntimeError, match="unknown type"):
            format_chunk({"chunk_type": "bogus"})
