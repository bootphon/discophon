set quiet

[private]
@default:
    just --list

# Test
test:
    uv run --group test pytest

# Lint and format
fmt:
    uv run --dev ruff check
    uv run --dev ruff format
    uv run --dev tombi format
    uv run --dev typos

# Build documentation
docs:
    echo "TODO"

# Serve live documentation
serve:
    echo "TODO"
