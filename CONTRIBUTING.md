# Contributing to strands-agents-portkey

Thank you for your interest in contributing! This guide covers how to set up your development environment and run the standard checks.

## Quick setup

```bash
# 1. Install Hatch
pip install hatch

# 2. Install the git hooks (runs the same checks as CI on every commit)
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## Prerequisites

- Python 3.10 or later
- [Hatch](https://hatch.pypa.io/) — the project's build and environment manager

Install Hatch globally:

```bash
pip install hatch
```

## Development workflow

All common tasks are available as Hatch scripts. Hatch automatically creates and manages an isolated virtual environment with all required dependencies.

### Run unit tests

```bash
hatch run test
```

Runs `pytest` against `tests/unit/` with coverage reporting. Coverage must stay at 100%.

### Run linter

```bash
hatch run lint
```

Runs `ruff check` over `src/` and `tests/unit/`. All rules must pass before a PR is merged.

### Format code

```bash
hatch run fmt
```

Runs `ruff format` over `src/` and `tests/`. Run this before committing to avoid CI failures.

### Type checking

```bash
hatch run type-check
```

Runs `mypy` over `src/` with strict settings.

### Run everything at once

```bash
hatch run check
```

Equivalent to `lint` + `type-check` + `test` in sequence. Use this before opening a pull request.

## Integration tests

Integration tests hit real provider APIs and require environment variables to be set.

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (via Portkey virtual key or direct)
export ANTHROPIC_API_KEY="sk-ant-..."

# Portkey API key
export PORTKEY_API_KEY="pk-..."

hatch run test-integ
```

Integration tests live in `tests/integration/` and are not run as part of CI — they must be executed locally or in a controlled environment with valid API credentials.

## Project structure

```
src/strands_portkey/
    __init__.py      # public exports: PortkeyModel, PortkeyConfig
    model.py         # PortkeyModel class
    _config.py       # PortkeyConfig Pydantic model
    _formatting.py   # request / response formatting helpers
    _errors.py       # error handling and exception mapping

tests/
    unit/                        # fast, dependency-free unit tests (100% coverage required)
    integration/                 # live API tests (requires credentials)
        test_client_args.py      # client_args passthrough (virtual key, provider slug, config, metadata)
        test_error_handling.py   # context overflow and throttle exception mapping
        test_model_params.py     # max_tokens, temperature
        test_multimodal.py       # image input
        test_parallel_tool_use.py # parallel tool calls and multi-result round-trips
        test_reasoning.py        # reasoning_content delta events
        test_streaming.py        # text streaming, event order, usage metadata, multi-turn
        test_structured_output.py # Pydantic structured output
        test_system_prompt.py    # system prompt string and content block variants
        test_tool_choice.py      # auto / any / specific tool_choice
        test_tool_use.py         # single tool call and tool result round-trip
```

## Code style

- Line length: 120 characters
- Docstrings: Google style (`convention = "google"` in ruff)
- All public functions and classes must have docstrings
- Imports are sorted by `ruff` (isort-compatible)

Run `hatch run fmt` to auto-format, then `hatch run lint` to catch anything remaining.

## Opening a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, add tests, and ensure `hatch run check` passes cleanly.
3. Open a pull request against `main` and fill in the PR template.
4. Tag `@roh26it` and `@narengogi` for review.
