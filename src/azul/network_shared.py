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
from typing import TYPE_CHECKING, Final, TypeAlias

from libcomponent.base_io import StructFormat
from libcomponent.buffer import Buffer
from mypy_extensions import u8
from numpy import int8, zeros

if TYPE_CHECKING:
    from numpy.typing import NDArray


ADVERTISEMENT_IP: Final = "224.0.2.60"
ADVERTISEMENT_PORT: Final = 4445

DEFAULT_PORT: Final = 31613

Pos: TypeAlias = tuple[u8, u8]


def encode_tile_count(tile_color: int, tile_count: int) -> Buffer:
    """Return buffer from tile color and count."""
    buffer = Buffer()

    buffer.write_value(StructFormat.UBYTE, tile_color)
    buffer.write_value(StructFormat.UBYTE, tile_count)

    return buffer


def decode_tile_count(buffer: Buffer) -> tuple[int, int]:
    """Read and return tile color and count from buffer."""
    tile_color = buffer.read_value(StructFormat.UBYTE)
    tile_count = buffer.read_value(StructFormat.UBYTE)

    return (tile_color, tile_count)


def encode_numeric_uint8_counter(counter: Counter[int]) -> Buffer:
    """Return buffer from uint8 counter."""
    buffer = Buffer()

    buffer.write_value(StructFormat.UBYTE, len(counter))
    for key, value in counter.items():
        assert isinstance(key, int)
        assert value >= 0
        buffer.extend(encode_tile_count(key, value))

    return buffer


def decode_numeric_uint8_counter(buffer: Buffer) -> Counter[int]:
    """Read and return uint8 counter from buffer."""
    data: dict[int, int] = {}

    pair_count = buffer.read_value(StructFormat.UBYTE)
    for _ in range(pair_count):
        key, value = decode_tile_count(buffer)
        assert key not in data
        data[key] = value

    return Counter(data)


def encode_int8_array(array: NDArray[int8]) -> Buffer:
    """Return buffer from int8 array flat values."""
    buffer = Buffer()

    for value in array.flat:
        buffer.write_value(StructFormat.BYTE, int(value))

    return buffer


def decode_int8_array(buffer: Buffer, size: tuple[int, ...]) -> NDArray[int8]:
    """Return flattened int8 array from buffer."""
    array = zeros(size, dtype=int8)

    for index in range(array.size):
        array.flat[index] = buffer.read_value(StructFormat.BYTE)

    return array


def encode_cursor_location(scaled_location: tuple[int, int]) -> bytes:
    """Return buffer from cursor location."""
    x, y = scaled_location
    position = ((x & 0xFFF) << 12) | (y & 0xFFF)
    return (position & 0xFFFFFF).to_bytes(3)


def decode_cursor_location(buffer: bytes | bytearray) -> tuple[int, int]:
    """Return cursor location from buffer."""
    value = int.from_bytes(buffer) & 0xFFFFFF
    x = (value >> 12) & 0xFFF
    y = value & 0xFFF
    return (x, y)


class ClientBoundEvents(IntEnum):
    """Client bound event IDs."""

    encryption_request = 0
    callback_ping = auto()
    initial_config = auto()
    playing_as = auto()
    game_over = auto()
    board_data = auto()
    pattern_data = auto()
    factory_data = auto()
    cursor_data = auto()
    table_data = auto()
    cursor_movement_mode = auto()
    current_turn_change = auto()
    cursor_position = auto()


class ServerBoundEvents(IntEnum):
    """Server bound event IDs."""

    encryption_response = 0
    factory_clicked = auto()
    cursor_location = auto()
    pattern_row_clicked = auto()
