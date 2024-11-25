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


from enum import IntEnum, auto
from typing import Final, NamedTuple, TypeAlias

from mypy_extensions import u8

ADVERTISEMENT_IP: Final = "224.0.2.60"
ADVERTISEMENT_PORT: Final = 4445

DEFAULT_PORT: Final = 31613

Pos: TypeAlias = tuple[u8, u8]


class TickEventData(NamedTuple):
    """Tick Event Data."""

    time_passed: float
    fps: float


class ClientBoundEvents(IntEnum):
    """Client bound event IDs."""

    encryption_request = 0
    callback_ping = auto()
    initial_config = auto()
    playing_as = auto()
    create_piece = auto()
    select_piece = auto()
    create_tile = auto()
    delete_tile = auto()
    animation_state = auto()
    delete_piece_animation = auto()
    update_piece_animation = auto()
    move_piece_animation = auto()
    action_complete = auto()
    game_over = auto()


class ServerBoundEvents(IntEnum):
    """Server bound event IDs."""

    encryption_response = 0
    select_piece = auto()
    select_tile = auto()
