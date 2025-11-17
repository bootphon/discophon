import importlib.resources
import json


def phone_to_phoneme() -> dict[str, dict[str, str]]:
    with importlib.resources.path("discophon.builder.rules", "rules.json") as fspath:
        return json.loads(fspath.read_text(encoding="utf-8"))
