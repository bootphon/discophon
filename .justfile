set quiet

[private]
@default:
    just --list

# Test
test:
    uv run --group test pytest

# Lint and format
lint:
    uv run --dev ruff check
    uv run --dev ruff format
    uv run --dev tombi format
    uv run --dev typos
    uv run --dev ty check src
