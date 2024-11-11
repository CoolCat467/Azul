"""Language file handler."""

from __future__ import annotations

# Programmed by CoolCat467

__title__ = "lang"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import json
from functools import cache
from os.path import exists, join


def load_json(filename: str) -> dict[str, str]:
    """Return json data loaded from filename."""
    with open(filename, encoding="utf-8") as loaded:
        data = json.load(loaded)
    assert isinstance(data, dict)
    return data


@cache
def load_lang(name: str) -> dict[str, str] | None:
    """Return full data for language with given name."""
    filename = join("lang", f"{name}.json")
    if not exists(filename):
        return None
    return load_json(filename)
