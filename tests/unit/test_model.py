"""Unit tests for PortkeyModel class (init, config, format_request, stream, structured_output)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

from strands_portkey.model import PortkeyModel

# --- Fixtures ---


@pytest.fixture
def model(portkey_client_args, portkey_model_config):
    """Create a PortkeyModel with mocked client."""
    with patch("strands_portkey.model.AsyncPortkey") as mock_portkey_cls:
        mock_client = MagicMock()
        mock_portkey_cls.return_value = mock_client
        m = PortkeyModel(client_args=portkey_client_args, **portkey_model_config)
        m._mock_client_cls = mock_portkey_cls
        return m


@pytest.fixture
def model_no_params():
    """Create a PortkeyModel with minimal config."""
    with patch("strands_portkey.model.AsyncPortkey"):
        return PortkeyModel(model_id="gpt-4o-mini")


# --- Helpers ---


def make_stream_chunk(delta_content=None, delta_tool_calls=None, finish_reason=None, reasoning_content=None):
    """Build a mock streaming chunk matching OpenAI format."""
    delta = MagicMock()
    delta.content = delta_content
    delta.tool_calls = delta_tool_calls

    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    else:
        # Make hasattr return False for reasoning_content
        del delta.reasoning_content

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def make_usage_chunk(prompt_tokens=10, completion_tokens=20, total_tokens=30):
    """Build a mock usage-only chunk (final event)."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = usage
    return chunk


def make_tool_call_delta(index=0, tool_id="tool-123", name="calculator", arguments='{"x": 1}'):
    """Build a mock tool call delta."""
    func = MagicMock()
    func.name = name
    func.arguments = arguments

    tc = MagicMock()
    tc.index = index
    tc.id = tool_id
    tc.function = func
    return tc


async def collect_stream(async_gen):
    """Collect all events from an async generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


# =============================================================================
# 1. Initialization & Configuration
# =============================================================================


class TestInit:
    def test_init_with_client_args(self, portkey_client_args):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            PortkeyModel(client_args=portkey_client_args, model_id="gpt-4o")
            mock_cls.assert_called_once_with(**portkey_client_args)

    def test_init_without_client_args(self):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            PortkeyModel(model_id="gpt-4o")
            mock_cls.assert_called_once_with()

    def test_init_stores_config(self, model):
        config = model.get_config()
        assert config["model_id"] == "gpt-4o"
        assert config["params"]["temperature"] == 0.7
        assert config["params"]["max_tokens"] == 1000

    def test_init_with_portkey_specific_args(self):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            PortkeyModel(
                client_args={
                    "api_key": "pk-key",
                    "virtual_key": "vk-key",
                    "config": "cfg-xxx",
                    "trace_id": "trace-1",
                    "metadata": {"env": "test"},
                },
                model_id="claude-sonnet-4-20250514",
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["api_key"] == "pk-key"
            assert call_kwargs["virtual_key"] == "vk-key"
            assert call_kwargs["config"] == "cfg-xxx"
            assert call_kwargs["trace_id"] == "trace-1"

    def test_init_with_provider_slug(self):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            PortkeyModel(
                client_args={
                    "api_key": "pk-key",
                    "provider": "openai",
                    "Authorization": "Bearer sk-xxx",
                },
                model_id="gpt-4o",
            )
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["provider"] == "openai"


# =============================================================================
# 2. Configuration
# =============================================================================


class TestConfig:
    def test_get_config(self, model):
        config = model.get_config()
        assert config["model_id"] == "gpt-4o"

    def test_update_config(self, model):
        model.update_config(model_id="gpt-4o-mini")
        assert model.get_config()["model_id"] == "gpt-4o-mini"

    def test_update_config_params(self, model):
        model.update_config(params={"temperature": 0.1})
        assert model.get_config()["params"]["temperature"] == 0.1

    def test_update_config_preserves_other_fields(self, model):
        model.update_config(params={"temperature": 0.1})
        assert model.get_config()["model_id"] == "gpt-4o"

    def test_config_without_params(self, model_no_params):
        config = model_no_params.get_config()
        assert config["model_id"] == "gpt-4o-mini"
        assert "params" not in config


# =============================================================================
# 3. Request Formatting (instance method — uses self.config)
# =============================================================================


class TestFormatRequest:
    def test_basic_request(self, model):
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]
        result = model.format_request(messages)
        assert result["model"] == "gpt-4o"
        assert result["stream"] is True
        assert result["stream_options"] == {"include_usage": True}
        assert len(result["messages"]) == 1
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000

    def test_request_with_tools(self, model):
        messages = [{"role": "user", "content": [{"text": "Calculate 2+2"}]}]
        tool_specs = [
            {
                "name": "calculator",
                "description": "Performs calculations",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    }
                },
            }
        ]
        result = model.format_request(messages, tool_specs=tool_specs)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "calculator"
        assert result["tools"][0]["function"]["description"] == "Performs calculations"

    def test_request_with_system_prompt(self, model):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = model.format_request(messages, system_prompt="Be concise")
        sys_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "Be concise"

    def test_request_without_tools(self, model):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = model.format_request(messages)
        assert "tools" not in result

    def test_request_without_params(self, model_no_params):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = model_no_params.format_request(messages)
        assert result["model"] == "gpt-4o-mini"
        assert "temperature" not in result

    def test_empty_tool_specs_excluded(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        assert "tools" not in m.format_request(messages, tool_specs=[])

    def test_none_tool_specs_excluded(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        assert "tools" not in m.format_request(messages, tool_specs=None)

    def test_tool_choice_included_when_tools_present(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        tool_specs = [
            {
                "name": "calc",
                "description": "Calculator",
                "inputSchema": {"json": {"type": "object", "properties": {"expr": {"type": "string"}}}},
            }
        ]
        result = m.format_request(messages, tool_specs=tool_specs, tool_choice={"any": {}})
        assert "tool_choice" in result
        assert result["tool_choice"] == "required"

    def test_tool_choice_excluded_without_tools(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = m.format_request(messages, tool_choice={"auto": {}})
        assert "tool_choice" not in result

    def test_tool_choice_excluded_when_none(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        tool_specs = [
            {
                "name": "calc",
                "description": "Calculator",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        ]
        result = m.format_request(messages, tool_specs=tool_specs, tool_choice=None)
        assert "tool_choice" not in result

    def test_system_prompt_content_in_request(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = m.format_request(messages, system_prompt_content=[{"text": "You are helpful."}])
        sys_msgs = [msg for msg in result["messages"] if msg["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "You are helpful."

    def test_system_prompt_content_overrides_system_prompt(self):
        with patch("strands_portkey.model.AsyncPortkey"):
            m = PortkeyModel(model_id="gpt-4o")
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        result = m.format_request(
            messages,
            system_prompt="This should be ignored",
            system_prompt_content=[{"text": "This takes precedence."}],
        )
        sys_msgs = [msg for msg in result["messages"] if msg["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "This takes precedence."


# =============================================================================
# 4. Streaming
# =============================================================================


class TestStream:
    @pytest.fixture
    def stream_model(self):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            return PortkeyModel(client_args={"api_key": "test"}, model_id="gpt-4o")

    @pytest.mark.asyncio
    async def test_simple_text_stream(self, stream_model):
        """Basic text streaming produces the correct event sequence."""
        chunks = [
            make_stream_chunk(delta_content="Hello"),
            make_stream_chunk(delta_content=" world"),
            make_stream_chunk(finish_reason="stop"),
        ]
        usage_chunk = make_usage_chunk(10, 5, 15)

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        events = await collect_stream(stream_model.stream(messages))

        event_types = [list(e.keys())[0] for e in events]
        assert event_types[0] == "messageStart"
        assert event_types[1] == "contentBlockStart"

        text_deltas = [e for e in events if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]]
        assert len(text_deltas) == 2
        assert text_deltas[0]["contentBlockDelta"]["delta"]["text"] == "Hello"
        assert text_deltas[1]["contentBlockDelta"]["delta"]["text"] == " world"

        assert {"contentBlockStop": {}} in events
        assert {"messageStop": {"stopReason": "end_turn"}} in events

        meta_events = [e for e in events if "metadata" in e]
        assert len(meta_events) == 1
        assert meta_events[0]["metadata"]["usage"]["inputTokens"] == 10

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, stream_model):
        """Tool use streaming produces correct tool content blocks."""
        tc1 = make_tool_call_delta(index=0, tool_id="t1", name="calc", arguments='{"expr":')
        tc2 = make_tool_call_delta(index=0, tool_id=None, name=None, arguments='"2+2"}')

        chunks = [
            make_stream_chunk(delta_content="Let me calculate."),
            make_stream_chunk(delta_tool_calls=[tc1]),
            make_stream_chunk(delta_tool_calls=[tc2]),
            make_stream_chunk(finish_reason="tool_calls"),
        ]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "What is 2+2?"}]}]
        tool_specs = [
            {
                "name": "calc",
                "description": "Calculator",
                "inputSchema": {"json": {"type": "object", "properties": {"expr": {"type": "string"}}}},
            }
        ]
        events = await collect_stream(stream_model.stream(messages, tool_specs=tool_specs))

        tool_starts = [
            e
            for e in events
            if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "calc"
        assert {"messageStop": {"stopReason": "tool_use"}} in events

    @pytest.mark.asyncio
    async def test_stream_with_reasoning(self, stream_model):
        """Reasoning content is forwarded as reasoningContent events."""
        chunks = [
            make_stream_chunk(reasoning_content="Let me think..."),
            make_stream_chunk(delta_content="The answer is 42."),
            make_stream_chunk(finish_reason="stop"),
        ]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Think about this"}]}]
        events = await collect_stream(stream_model.stream(messages))

        reasoning_events = [
            e
            for e in events
            if "contentBlockDelta" in e and "reasoningContent" in e.get("contentBlockDelta", {}).get("delta", {})
        ]
        assert len(reasoning_events) == 1
        assert reasoning_events[0]["contentBlockDelta"]["delta"]["reasoningContent"]["text"] == "Let me think..."

    @pytest.mark.asyncio
    async def test_stream_max_tokens(self, stream_model):
        chunks = [
            make_stream_chunk(delta_content="Partial response"),
            make_stream_chunk(finish_reason="length"),
        ]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Write a long essay"}]}]
        events = await collect_stream(stream_model.stream(messages))

        assert {"messageStop": {"stopReason": "max_tokens"}} in events

    @pytest.mark.asyncio
    async def test_stream_skips_empty_choices(self, stream_model):
        """Chunks with no choices are skipped."""
        empty_chunk = MagicMock()
        empty_chunk.choices = []

        normal_chunk = make_stream_chunk(delta_content="Hi", finish_reason="stop")
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            yield empty_chunk
            yield normal_chunk
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        events = await collect_stream(stream_model.stream(messages))

        text_events = [e for e in events if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]]
        assert len(text_events) == 1

    @pytest.mark.asyncio
    async def test_stream_multiple_tool_calls(self, stream_model):
        """Multiple concurrent tool calls all produce content blocks."""
        tc1 = make_tool_call_delta(index=0, tool_id="t1", name="search", arguments='{"q": "hello"}')
        tc2 = make_tool_call_delta(index=1, tool_id="t2", name="calc", arguments='{"x": 1}')

        chunks = [
            make_stream_chunk(delta_tool_calls=[tc1, tc2]),
            make_stream_chunk(finish_reason="tool_calls"),
        ]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Search and calculate"}]}]
        events = await collect_stream(stream_model.stream(messages))

        tool_starts = [
            e
            for e in events
            if "contentBlockStart" in e and "toolUse" in e.get("contentBlockStart", {}).get("start", {})
        ]
        assert len(tool_starts) == 2

    @pytest.mark.asyncio
    async def test_stream_no_usage(self, stream_model):
        """Streaming without usage data produces no metadata event."""
        chunks = [make_stream_chunk(delta_content="Hi", finish_reason="stop")]
        final_chunk = MagicMock()
        final_chunk.choices = []
        final_chunk.usage = None

        async def mock_stream():
            for c in chunks:
                yield c
            yield final_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        events = await collect_stream(stream_model.stream(messages))

        assert len([e for e in events if "metadata" in e]) == 0

    @pytest.mark.asyncio
    async def test_stream_passes_system_prompt(self, stream_model):
        chunks = [make_stream_chunk(delta_content="Ok", finish_reason="stop")]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        await collect_stream(stream_model.stream(messages, system_prompt="Be helpful"))

        call_kwargs = stream_model.client.chat.completions.create.call_args[1]
        sys_msgs = [m for m in call_kwargs["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_stream_passes_tool_choice(self, stream_model):
        chunks = [make_stream_chunk(delta_content="Ok", finish_reason="stop")]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        tool_specs = [
            {
                "name": "calc",
                "description": "Calculator",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        ]
        await collect_stream(stream_model.stream(messages, tool_specs=tool_specs, tool_choice={"any": {}}))

        call_kwargs = stream_model.client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_stream_passes_system_prompt_content(self, stream_model):
        chunks = [make_stream_chunk(delta_content="Ok", finish_reason="stop")]
        usage_chunk = make_usage_chunk()

        async def mock_stream():
            for c in chunks:
                yield c
            yield usage_chunk

        stream_model.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        await collect_stream(stream_model.stream(messages, system_prompt_content=[{"text": "Be concise."}]))

        call_kwargs = stream_model.client.chat.completions.create.call_args[1]
        sys_msgs = [m for m in call_kwargs["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "Be concise."


# =============================================================================
# 5. Structured Output
# =============================================================================


class TestStructuredOutput:
    @pytest.fixture
    def so_model(self):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            return PortkeyModel(client_args={"api_key": "test"}, model_id="gpt-4o")

    @pytest.mark.asyncio
    async def test_structured_output_success(self, so_model):
        class WeatherInfo(BaseModel):
            city: str
            temperature: float
            unit: str

        parsed = WeatherInfo(city="NYC", temperature=72.0, unit="F")
        mock_choice = MagicMock()
        mock_choice.message.parsed = parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        so_model.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": [{"text": "What's the weather in NYC?"}]}]
        events = await collect_stream(so_model.structured_output(WeatherInfo, messages))

        assert len(events) == 1
        assert events[0]["output"].city == "NYC"
        assert events[0]["output"].temperature == 72.0

    @pytest.mark.asyncio
    async def test_structured_output_no_match(self, so_model):
        class MyModel(BaseModel):
            x: int

        mock_choice = MagicMock()
        mock_choice.message.parsed = "not a MyModel instance"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        so_model.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": [{"text": "test"}]}]
        with pytest.raises(ValueError, match="No valid structured output"):
            await collect_stream(so_model.structured_output(MyModel, messages))

    @pytest.mark.asyncio
    async def test_structured_output_multiple_choices_raises(self, so_model):
        class MyModel(BaseModel):
            x: int

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(), MagicMock()]

        so_model.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": [{"text": "test"}]}]
        with pytest.raises(ValueError, match="Multiple choices"):
            await collect_stream(so_model.structured_output(MyModel, messages))

    @pytest.mark.asyncio
    async def test_structured_output_passes_system_prompt(self, so_model):
        class Info(BaseModel):
            data: str

        parsed = Info(data="test")
        mock_choice = MagicMock()
        mock_choice.message.parsed = parsed
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        so_model.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": [{"text": "test"}]}]
        await collect_stream(so_model.structured_output(Info, messages, system_prompt="Be precise"))

        call_kwargs = so_model.client.beta.chat.completions.parse.call_args[1]
        sys_msgs = [m for m in call_kwargs["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1


# =============================================================================
# 6. Error Handling (via stream — exercises handle_api_error integration)
# =============================================================================


class TestErrorHandling:
    @pytest.fixture
    def err_model(self):
        with patch("strands_portkey.model.AsyncPortkey") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            return PortkeyModel(client_args={"api_key": "test"}, model_id="gpt-4o")

    @pytest.mark.asyncio
    async def test_context_window_overflow(self, err_model):
        err_model.client.chat.completions.create = AsyncMock(
            side_effect=Exception("maximum context length exceeded for model gpt-4o")
        )
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        with pytest.raises(ContextWindowOverflowException):
            await collect_stream(err_model.stream(messages))

    @pytest.mark.asyncio
    async def test_context_window_prompt_too_long(self, err_model):
        err_model.client.chat.completions.create = AsyncMock(side_effect=Exception("prompt is too long: 200000 tokens"))
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        with pytest.raises(ContextWindowOverflowException):
            await collect_stream(err_model.stream(messages))

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, err_model):
        err_model.client.chat.completions.create = AsyncMock(side_effect=Exception("rate limit exceeded"))
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        with pytest.raises(ModelThrottledException):
            await collect_stream(err_model.stream(messages))

    @pytest.mark.asyncio
    async def test_overloaded_error(self, err_model):
        err_model.client.chat.completions.create = AsyncMock(side_effect=Exception("Overloaded"))
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        with pytest.raises(ModelThrottledException):
            await collect_stream(err_model.stream(messages))

    @pytest.mark.asyncio
    async def test_429_error(self, err_model):
        err_model.client.chat.completions.create = AsyncMock(
            side_effect=Exception("Error code: 429 - Too Many Requests")
        )
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        with pytest.raises(ModelThrottledException):
            await collect_stream(err_model.stream(messages))

    @pytest.mark.asyncio
    async def test_unrelated_error_passes_through(self, err_model):
        err_model.client.chat.completions.create = AsyncMock(side_effect=ValueError("Something else broke"))
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]
        with pytest.raises(ValueError, match="Something else broke"):
            await collect_stream(err_model.stream(messages))
