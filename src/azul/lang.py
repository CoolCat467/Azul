#!/usr/bin/env python3
# Language file handler

"""Language file handler."""

# Programmed by CoolCat467

__title__ = "lang"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import json
from functools import cache
from os.path import exists, join


def load_json(filename: str) -> dict:
    """Return json data loaded from filename."""
    with open(filename, encoding="utf-8") as loaded:
        data = json.load(loaded)
        loaded.close()
    return data


@cache
def load_lang(name: str) -> dict | None:
    """Return full data for language with given name."""
    filename = join("lang", name + ".json")
    if not exists(filename):
        return None
    return load_json(filename)


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
##    run()
