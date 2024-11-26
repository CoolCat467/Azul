"""Shared network code."""

# Programmed by CoolCat467

from __future__ import annotations

# Copyright (C) 2023-2024  CoolCat467
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__title__ = "Network Shared"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"


from collections import Counter
from enum import IntEnum, auto
from typing import Final, TypeAlias

from libcomponent.base_io import StructFormat
from libcomponent.buffer import Buffer
from mypy_extensions import u8

ADVERTISEMENT_IP: Final = "224.0.2.60"
ADVERTISEMENT_PORT: Final = 4445

DEFAULT_PORT: Final = 31613

Pos: TypeAlias = tuple[u8, u8]


def encode_numeric_uint8_counter(counter: Counter[int]) -> Buffer:
    """Return buffer from uint8 counter (both keys and values)."""
    buffer = Buffer()

    for key, value in counter.items():
        assert isinstance(key, int)
        buffer.write_value(StructFormat.UBYTE, key)
        buffer.write_value(StructFormat.UBYTE, value)
    return buffer


def decode_numeric_uint8_counter(buffer: Buffer) -> Counter[int]:
    """Return buffer from uint8 counter (both keys and values)."""
    data: dict[int, int] = {}

    for _ in range(0, len(buffer), 2):
        key = buffer.read_value(StructFormat.UBYTE)
        value = buffer.read_value(StructFormat.UBYTE)
        assert key not in data
        data[key] = value

    return Counter(data)


class ClientBoundEvents(IntEnum):
    """Client bound event IDs."""

    encryption_request = 0
    callback_ping = auto()
    initial_config = auto()
    playing_as = auto()
    game_over = auto()


class ServerBoundEvents(IntEnum):
    """Server bound event IDs."""

    encryption_response = 0
