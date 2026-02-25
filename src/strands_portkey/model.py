"""Portkey model provider for Strands Agents.

- Docs: https://portkey.ai/docs
"""

import logging
from typing import Any, AsyncGenerator, Optional, Type, TypeVar, Union, cast

from portkey_ai import AsyncPortkey
from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from typing_extensions import override

from ._config import PortkeyConfig
from ._errors import handle_api_error
from ._formatting import format_chunk, format_request_messages, format_tool_choice

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class PortkeyModel(Model):
    """Portkey model provider implementation.

    Routes requests through Portkey's AI gateway, providing unified access to 250+ LLMs
    with built-in fallbacks, load balancing, caching, and observability.
    """

    def __init__(
        self,
        client_args: Optional[dict[str, Any]] = None,
        **model_config: Any,
    ) -> None:
        """Initialize the Portkey model provider.

        Args:
            client_args: Arguments for the AsyncPortkey client. Common options:
                - api_key: Portkey API key.
                - virtual_key: Virtual key for provider authentication.
                - provider: Provider slug (e.g., "openai", "anthropic").
                - config: Portkey config ID or inline config dict for routing strategies.
                - trace_id: Trace ID for request tracing.
                - metadata: Metadata dict for observability.
                - Authorization: Direct provider API key (when using provider slug).
            **model_config: Fields for :class:`PortkeyConfig` (``model_id``, ``params``).

        Raises:
            pydantic.ValidationError: If ``model_config`` fails validation
                (e.g., missing or blank ``model_id``).
        """
        self.config = PortkeyConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = AsyncPortkey(**client_args)

    @override
    def update_config(self, **model_config: Any) -> None:
        """Update the model configuration.

        Args:
            **model_config: Fields to override in the current :class:`PortkeyConfig`.

        Raises:
            pydantic.ValidationError: If the resulting config fails validation.
        """
        self.config = PortkeyConfig(**{**self.config.model_dump(), **model_config})

    @override
    def get_config(self) -> PortkeyConfig:
        """Return the current validated configuration.

        Returns:
            The current :class:`PortkeyConfig` instance.
        """
        return self.config

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: Optional[ToolChoice] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
    ) -> dict[str, Any]:
        """Format a streaming chat request.

        Args:
            messages: List of message objects.
            tool_specs: Tool specifications available to the model.
            system_prompt: System prompt to provide context.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks.

        Returns:
            Formatted request dict ready to be unpacked into the Portkey API call.
        """
        request: dict[str, Any] = {
            "messages": format_request_messages(messages, system_prompt, system_prompt_content),
            "model": self.config.model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
            **cast(dict[str, Any], self.config.params or {}),
        }

        # Only include tools when there are actual tool specs — some providers
        # (e.g. Anthropic's OpenAI-compat endpoint) reject empty tools arrays.
        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs
            ]
            if tool_choice:
                request["tool_choice"] = format_tool_choice(tool_choice)

        return request

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: Optional[ToolChoice] = None,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
        invocation_state: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a conversation through Portkey.

        Args:
            messages: List of message objects.
            tool_specs: Tool specifications available to the model.
            system_prompt: System prompt to provide context.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks for advanced features like caching.
            invocation_state: Caller-provided state/context (unused, for interface compatibility).
            **kwargs: Additional keyword arguments.

        Yields:
            Formatted StreamEvent chunks.

        Raises:
            ContextWindowOverflowException: When the input exceeds the model's context window.
            ModelThrottledException: When the model service is throttling requests.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice, system_prompt_content)
        logger.debug("formatted request=<%s>", request)

        try:
            logger.debug("invoking model via Portkey")
            response = await self.client.chat.completions.create(**request)
        except Exception as e:
            handle_api_error(e)
            raise

        logger.debug("got response from model")
        yield format_chunk({"chunk_type": "message_start"})
        yield format_chunk({"chunk_type": "content_start", "data_type": "text"})

        tool_calls: dict[int, list[Any]] = {}

        async for event in response:  # type: ignore[union-attr]
            if not getattr(event, "choices", None):
                continue
            choice = event.choices[0]  # type: ignore[index]
            delta = choice.delta  # type: ignore[union-attr]

            if delta.content:  # type: ignore[union-attr]
                yield format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": delta.content})  # type: ignore[union-attr]

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:  # type: ignore[union-attr]
                yield format_chunk(
                    {
                        "chunk_type": "content_delta",
                        "data_type": "reasoning_content",
                        "data": delta.reasoning_content,  # type: ignore[union-attr]
                    }
                )

            for tool_call in delta.tool_calls or []:  # type: ignore[union-attr]
                tool_calls.setdefault(tool_call.index, []).append(tool_call)  # type: ignore[arg-type]

            if choice.finish_reason:
                break

        yield format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        for tool_deltas in tool_calls.values():
            yield format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})
            for tool_delta in tool_deltas:
                yield format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})
            yield format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

        yield format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

        async for event in response:  # type: ignore[union-attr]
            _ = event

        if event.usage:
            yield format_chunk({"chunk_type": "metadata", "data": event.usage})

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model via Portkey.

        Args:
            output_model: Pydantic model defining the expected output schema.
            prompt: The prompt messages.
            system_prompt: System prompt to provide context.
            **kwargs: Additional keyword arguments.

        Yields:
            Model events with the last being the structured output.
        """
        response = await self.client.beta.chat.completions.parse(
            model=self.config.model_id,
            messages=self.format_request(prompt, system_prompt=system_prompt)["messages"],
            response_format=output_model,
        )

        parsed: T | None = None
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the response.")

        for choice in response.choices:
            if isinstance(choice.message.parsed, output_model):
                parsed = choice.message.parsed
                break

        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid structured output was found in the response.")
