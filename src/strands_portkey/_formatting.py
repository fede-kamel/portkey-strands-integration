"""Request and response formatting utilities for the Portkey model provider."""

import json
from typing import Any, Optional, Union, cast

from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolUse


def format_request_message_content(content: ContentBlock) -> dict[str, Any]:
    """Format a content block for the OpenAI-compatible API.

    Args:
        content: Message content.

    Returns:
        Formatted content block.

    Raises:
        TypeError: If the content block type is unsupported.
    """
    if "image" in content:
        import base64
        import mimetypes

        mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
        image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8")

        return {
            "image_url": {
                "detail": "auto",
                "format": mime_type,
                "url": f"data:{mime_type};base64,{image_data}",
            },
            "type": "image_url",
        }

    if "text" in content:
        return {"text": content["text"], "type": "text"}

    raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")


def format_request_message_tool_call(tool_use: ToolUse) -> dict[str, Any]:
    """Format a tool call for the OpenAI-compatible API.

    Args:
        tool_use: Tool use requested by the model.

    Returns:
        Formatted tool call.
    """
    return {
        "function": {
            "arguments": json.dumps(tool_use["input"]),
            "name": tool_use["name"],
        },
        "id": tool_use["toolUseId"],
        "type": "function",
    }


def format_request_tool_message(tool_result: ToolResult) -> dict[str, Any]:
    """Format a tool result message for the OpenAI-compatible API.

    Args:
        tool_result: Tool result from a tool execution.

    Returns:
        Formatted tool message.
    """
    contents = cast(
        list[ContentBlock],
        [{"text": json.dumps(content["json"])} if "json" in content else content for content in tool_result["content"]],
    )

    return {
        "role": "tool",
        "tool_call_id": tool_result["toolUseId"],
        "content": [format_request_message_content(content) for content in contents],
    }


def format_request_messages(
    messages: Messages,
    system_prompt: Optional[str] = None,
    system_prompt_content: Optional[list[SystemContentBlock]] = None,
) -> list[dict[str, Any]]:
    """Format messages into the OpenAI-compatible format.

    Args:
        messages: List of message objects.
        system_prompt: System prompt to provide context.
        system_prompt_content: System prompt content blocks (takes precedence over system_prompt).

    Returns:
        Formatted messages array.
    """
    formatted_messages: list[dict[str, Any]]
    if system_prompt_content:
        system_text = " ".join(block.get("text", "") for block in system_prompt_content if "text" in block)
        formatted_messages = [{"role": "system", "content": system_text}] if system_text else []
    elif system_prompt:
        formatted_messages = [{"role": "system", "content": system_prompt}]
    else:
        formatted_messages = []

    for message in messages:
        contents = message["content"]

        formatted_contents = [
            format_request_message_content(content)
            for content in contents
            if not any(block_type in content for block_type in ["toolResult", "toolUse"])
        ]
        formatted_tool_calls = [
            format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
        ]
        formatted_tool_messages = [
            format_request_tool_message(content["toolResult"]) for content in contents if "toolResult" in content
        ]

        formatted_message = {
            "role": message["role"],
            "content": formatted_contents,
            **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
        }
        formatted_messages.append(formatted_message)
        formatted_messages.extend(formatted_tool_messages)

    return [message for message in formatted_messages if message["content"] or "tool_calls" in message]


def format_tool_choice(tool_choice: ToolChoice) -> Union[str, dict[str, Any]]:
    """Format a Strands ToolChoice into OpenAI-compatible tool_choice.

    Args:
        tool_choice: Strands tool choice specification.

    Returns:
        OpenAI-compatible tool_choice value.
    """
    if "auto" in tool_choice:
        return "auto"  # type: ignore[return-value]
    if "any" in tool_choice:
        return "required"  # type: ignore[return-value]
    if "tool" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["tool"]["name"]}}  # type: ignore[index,typeddict-item]
    return "auto"  # type: ignore[return-value]


def format_chunk(event: dict[str, Any]) -> StreamEvent:
    """Format a response event into a Strands StreamEvent.

    Args:
        event: A response event from the model.

    Returns:
        Formatted StreamEvent.

    Raises:
        RuntimeError: If chunk_type is not recognized.
    """
    match event["chunk_type"]:
        case "message_start":
            return {"messageStart": {"role": "assistant"}}

        case "content_start":
            if event["data_type"] == "tool":
                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": event["data"].function.name,
                                "toolUseId": event["data"].id,
                            }
                        }
                    }
                }
            return {"contentBlockStart": {"start": {}}}

        case "content_delta":
            if event["data_type"] == "tool":
                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}}
            if event["data_type"] == "reasoning_content":
                return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}
            return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

        case "content_stop":
            return {"contentBlockStop": {}}

        case "message_stop":
            match event["data"]:
                case "tool_calls":
                    return {"messageStop": {"stopReason": "tool_use"}}
                case "length":
                    return {"messageStop": {"stopReason": "max_tokens"}}
                case _:
                    return {"messageStop": {"stopReason": "end_turn"}}

        case "metadata":
            return {
                "metadata": {
                    "usage": {
                        "inputTokens": event["data"].prompt_tokens,
                        "outputTokens": event["data"].completion_tokens,
                        "totalTokens": event["data"].total_tokens,
                    },
                    "metrics": {
                        "latencyMs": 0,
                    },
                },
            }

        case _:
            raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")
