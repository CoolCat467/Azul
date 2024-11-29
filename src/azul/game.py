"""Azul board game clone, now on the computer."""

# Programmed by CoolCat467

from __future__ import annotations

# Copyright (C) 2018-2024  CoolCat467
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

__title__ = "Azul"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"
__version__ = "2.0.0"

import contextlib
import importlib
import math
import os
import sys
import time
import traceback
from collections import Counter
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeVar

import pygame
import trio
from libcomponent.component import (
    ComponentManager,
    Event,
    ExternalRaiseManager,
)
from libcomponent.network_utils import find_ip
from numpy import array, int8
from pygame.color import Color
from pygame.locals import (
    K_ESCAPE,
    KEYUP,
    QUIT,
    RESIZABLE,
    SRCALPHA,
    USEREVENT,
    WINDOWRESIZED,
)
from pygame.rect import Rect

from azul import element_list, objects, sprite
from azul.async_clock import Clock
from azul.client import GameClient, read_advertisements
from azul.network_shared import DEFAULT_PORT
from azul.server import GameServer
from azul.sound import SoundData, play_sound as base_play_sound
from azul.state import Tile
from azul.statemachine import AsyncState
from azul.tools import (
    lerp_color,
)
from azul.vector import Vector2

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
        Generator,
        Iterable,
        Sequence,
    )

    from numpy.typing import NDArray
    from typing_extensions import TypeVarTuple

    P = TypeVarTuple("P")

T = TypeVar("T")
RT = TypeVar("RT")

SCREEN_SIZE = (650, 600)
VSYNC = True

FPS: Final = 48

# Colors
BLACK: Final = (0, 0, 0)
BLUE: Final = (15, 15, 255)
GREEN: Final = (0, 255, 0)
CYAN: Final = (0, 255, 255)
RED: Final = (255, 0, 0)
MAGENTA: Final = (255, 0, 255)
YELLOW: Final = (255, 220, 0)
WHITE: Final = (255, 255, 255)
GREY = (170, 170, 170)
ORANGE = (255, 128, 0)
DARKGREEN = (0, 128, 0)
DARKCYAN = (0, 128, 128)


if globals().get("__file__") is None:
    import importlib

    __file__ = str(
        Path(importlib.import_module("azul.data").__path__[0]).parent
        / "game.py",
    )

ROOT_FOLDER: Final = Path(__file__).absolute().parent
DATA_FOLDER: Final = ROOT_FOLDER / "data"
FONT_FOLDER: Final = ROOT_FOLDER / "fonts"

# Game stuff
# Tiles
TILECOUNT = 100
REGTILECOUNT = 5
tile_colors = (BLUE, YELLOW, RED, BLACK, CYAN, (WHITE, BLUE))
TILESYMBOLS = (
    ("*", WHITE),
    ("X", BLACK),
    ("+", BLACK),
    ("?", YELLOW),
    ("&", ORANGE),
    ("1", BLUE),
)


TILESIZE = 15

# Colors
BACKGROUND = (0, 192, 16)
TILEDEFAULT = ORANGE
SCORECOLOR = BLACK
PATSELECTCOLOR = DARKGREEN
BUTTON_TEXT_COLOR = DARKCYAN
BUTTON_TEXT_OUTLINE = BLACK
BUTTONBACKCOLOR = WHITE
GREYSHIFT = 0.75  # 0.65

# Font
FONT: Final = FONT_FOLDER / "VeraSerif.ttf"  # "RuneScape-UF-Regular.ttf"
SCOREFONTSIZE = 30
BUTTONFONTSIZE = 60

SOUND_LOOKUP: Final = {
    "delete_piece": "pop.mp3",
    "piece_move": "slide.mp3",
    "piece_update": "ding.mp3",
    "game_won": "newthingget.ogg",
    "button_click": "select.mp3",
    "tick": "tick.mp3",
}
SOUND_DATA: Final = {
    "delete_piece": SoundData(
        volume=50,
    ),
}


@lru_cache
def make_square_surf(
    color: (
        pygame.color.Color
        | int
        | str
        | tuple[int, int, int]
        | tuple[int, int, int, int]
        | Sequence[int]
    ),
    size: int,
) -> pygame.surface.Surface:
    """Return a surface of a square of given color and size."""
    s = int(size)
    surf = pygame.Surface((s, s))
    surf.fill(color)
    return surf


def play_sound(
    sound_name: str,
) -> tuple[pygame.mixer.Sound, int | float]:
    """Play sound effect."""
    sound_filename = SOUND_LOOKUP.get(sound_name)
    if sound_filename is None:
        raise RuntimeError(f"Error: Sound with ID `{sound_name}` not found.")
    sound_data = SOUND_DATA.get(sound_name, SoundData())

    return base_play_sound(
        DATA_FOLDER / sound_filename,
        sound_data,
    )


def outline_rectangle(
    surface: pygame.surface.Surface,
    color: (
        pygame.color.Color
        | int
        | str
        | tuple[int, int, int]
        | tuple[int, int, int, int]
        | Sequence[int]
    ),
    percent: float = 0.1,
) -> pygame.surface.Surface:
    """Return a surface after adding an outline of given color. Percentage is how big the outline is."""
    w, h = surface.get_size()
    inside_surf = pygame.transform.scale(
        surface.copy(),
        (round(w * (1 - percent)), round(h * (1 - percent))),
    )
    surface.fill(color)
    surface.blit(
        inside_surf,
        (math.floor(w * percent / 2), math.floor(h * percent / 2)),
    )
    return surface


def auto_crop_clear(
    surface: pygame.surface.Surface,
    clear: pygame.color.Color | None = None,
) -> pygame.surface.Surface:
    """Remove unneccicary pixels from image."""
    if clear is None:
        clear = pygame.color.Color(0, 0, 0, 0)
    surface = surface.convert_alpha()
    w, h = surface.get_size()
    surface.lock()

    def find_end(
        iterfunc: Callable[[int], Iterable[pygame.color.Color]],
        rangeobj: Iterable[int],
    ) -> int:
        for x in rangeobj:
            if not all(y == clear for y in iterfunc(x)):
                return x
        return x

    def column(x: int) -> Generator[pygame.color.Color, None, None]:
        return (surface.get_at((x, y)) for y in range(h))

    def row(y: int) -> Generator[pygame.color.Color, None, None]:
        return (surface.get_at((x, y)) for x in range(w))

    leftc = find_end(column, range(w))
    rightc = find_end(column, range(w - 1, -1, -1))
    topc = find_end(row, range(h))
    floorc = find_end(row, range(h - 1, -1, -1))
    surface.unlock()
    dim = pygame.rect.Rect(leftc, topc, rightc - leftc, floorc - topc)
    return surface.subsurface(dim)


@lru_cache
def get_tile_color(
    tile_color: int,
    greyshift: float = GREYSHIFT,
) -> tuple[int, int, int] | tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the color a given tile should be."""
    if tile_color < 0:
        if tile_color == Tile.blank:
            return GREY
        color = tile_colors[abs(tile_color + 1)]
        assert len(color) == 3
        r, g, b = lerp_color(color, GREY, greyshift)
        return int(r), int(g), int(b)
    if tile_color < 5:
        return tile_colors[tile_color]
    raise ValueError("Invalid tile color")


@lru_cache
def get_tile_symbol_and_color(
    tile_color: int,
    greyshift: float = GREYSHIFT,
) -> tuple[str, tuple[int, int, int]]:
    """Return the color a given tile should be."""
    if tile_color < 0:
        if tile_color == Tile.blank:
            return " ", GREY
        symbol, scolor = TILESYMBOLS[abs(tile_color + 1)]
        r, g, b = lerp_color(scolor, GREY, greyshift)
        return symbol, (int(r), int(g), int(b))
    if tile_color <= 5:
        return TILESYMBOLS[tile_color]
    raise ValueError("Invalid tile color")


def add_symbol_to_tile_surf(
    surf: pygame.surface.Surface,
    tilecolor: int,
    tilesize: int,
    greyshift: float = GREYSHIFT,
) -> None:
    """Add symbol to tile surface."""
    font = FONT_FOLDER / "VeraSerif.ttf"
    symbol, scolor = get_tile_symbol_and_color(tilecolor, greyshift)
    pyfont = pygame.font.Font(font, math.floor(math.sqrt(tilesize**2 * 2)) - 1)

    symbolsurf = pyfont.render(symbol, True, scolor)
    symbolsurf = auto_crop_clear(symbolsurf)

    width, height = symbolsurf.get_size()
    scale_factor = tilesize / height
    symbolsurf = pygame.transform.scale(
        symbolsurf,
        (width * scale_factor, height * scale_factor),
    )

    # symbolsurf = pygame.transform.scale(symbolsurf, (tilesize, tilesize))

    # sw, sh = symbolsurf.get_size()
    ##
    # w, h = surf.get_size()
    ##
    # x = w/2 - sw/2
    # y = h/2 - sh/2
    # b = (round(x), round(y))

    sw, sh = symbolsurf.get_rect().center
    w, h = surf.get_rect().center
    x = w - sw
    y = h - sh

    surf.blit(symbolsurf, (int(x), int(y)))


# surf.blit(symbolsurf, (0, 0))


def get_tile_image(
    tile_color: int,
    tilesize: int,
    greyshift: float = GREYSHIFT,
    outline_size: float = 0.2,
) -> pygame.surface.Surface:
    """Return a surface of a given tile."""
    if tile_color < 5:
        color = get_tile_color(tile_color, greyshift)

    elif tile_color >= 5:
        color_data = tile_colors[tile_color]
        assert len(color_data) == 2
        color, outline = color_data
        surf = outline_rectangle(
            make_square_surf(color, tilesize),
            outline,
            outline_size,
        )
        # Add tile symbol
        add_symbol_to_tile_surf(surf, tile_color, tilesize, greyshift)

        return surf
    surf = make_square_surf(color, tilesize)
    # Add tile symbol
    add_symbol_to_tile_surf(surf, tile_color, tilesize, greyshift)
    return surf


def get_tile_container_image(
    width_height: tuple[int, int],
    back: (
        pygame.color.Color
        | int
        | str
        | tuple[int, int, int]
        | tuple[int, int, int, int]
        | Sequence[int]
        | None
    ),
) -> pygame.surface.Surface:
    """Return a tile container image from a width and a height and a background color, and use a game's cache to help."""
    image = pygame.surface.Surface(width_height, flags=SRCALPHA)
    if back is not None:
        image.fill(back)
    else:
        image.fill((0, 0, 0, 0))
    return image


class TileRenderer(sprite.Sprite):
    """Base class for all objects that need to render tiles."""

    __slots__ = ("background", "tile_separation")
    greyshift = GREYSHIFT
    tile_size = TILESIZE

    def __init__(
        self,
        name: str,
        tile_separation: int | None = None,
        background: tuple[int, int, int] | None = TILEDEFAULT,
    ) -> None:
        """Initialize renderer."""
        super().__init__(name)

        if tile_separation is None:
            self.tile_separation = self.tile_size / 3.75
        else:
            self.tile_separation = tile_separation

        self.background = background

    def clear_image(
        self,
        tile_dimensions: tuple[int, int],
        extra: tuple[int, int] | None = None,
    ) -> None:
        """Reset self.image using tile_dimensions tuple and fills with self.background. Also updates self.width_height."""
        size = Vector2.from_iter(tile_dimensions)
        tile_full = self.tile_size + self.tile_separation
        size *= tile_full

        offset = Vector2(self.tile_separation, self.tile_separation)

        if extra is not None:
            offset += extra

        size += offset

        self.image = get_tile_container_image(
            round(size),
            self.background,
        )

    def blit_tile(
        self,
        tile_color: int,
        tile_location: tuple[int, int],
        offset: tuple[int, int] | None = None,
    ) -> None:
        """Blit the surface of a given tile object onto self.image at given tile location. It is assumed that all tile locations are xy tuples."""
        x, y = tile_location
        if offset is None:
            ox, oy = 0, 0
        else:
            ox, oy = offset

        ox += self.tile_separation
        oy += self.tile_separation

        surf = get_tile_image(tile_color, self.tile_size, self.greyshift)
        assert self.image is not None

        tile_full = self.tile_size + self.tile_separation

        self.image.blit(
            surf,
            (
                round(x * tile_full + ox),
                round(y * tile_full + oy),
            ),
        )

    def to_image_surface_location(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> Vector2:
        """Return screen location with respect to top left of image."""
        return Vector2.from_points(self.rect.topleft, screen_location)

    def get_tile_point(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> tuple[int, int] | None:
        """Return the xy choordinates of which tile intersects given a point or None."""
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.is_selected(screen_location):
            return None

        # Find out where screen point is in image locations
        # board x and y
        surface_pos = self.to_image_surface_location(screen_location)
        # Subtract separation boarder offset
        surface_pos -= (self.tile_separation, self.tile_separation)

        tile_full = self.tile_size + self.tile_separation

        # Get tile position and offset into that tile
        tile_position, offset = divmod(surface_pos, tile_full)
        for value in offset:
            # If in separation region, not selected
            if value > self.tile_size:
                return None
        # Otherwise, not in separation region, so we should be good
        return tile_position


class Cursor(TileRenderer):
    """Cursor TileRenderer.

    Registers following event handlers:
    - cursor_drag
    - cursor_reached_destination
    - cursor_set_destination
    - cursor_set_movement_mode

    Sometimes registered:
    - PygameMouseMotion
    """

    __slots__ = ("tiles",)
    greyshift = GREYSHIFT

    def __init__(self) -> None:
        """Initialize cursor with a game it belongs to."""
        super().__init__("Cursor", background=None)
        self.update_location_on_resize = True

        self.add_components(
            (
                sprite.MovementComponent(speed=800),
                sprite.TargetingComponent("cursor_reached_destination"),
            ),
        )

        # Stored in reverse render order
        self.tiles: list[int] = []

    def update_image(self) -> None:
        """Update self.image."""
        tile_count = len(self.tiles)
        self.clear_image((tile_count, 1))

        # Render in reverse order so keeping number one on end is easier
        for x in range(tile_count):
            self.blit_tile(self.tiles[tile_count - x - 1], (x, 0))
        if tile_count:
            self.dirty = 1
        self.visible = bool(tile_count)

    def bind_handlers(self) -> None:
        """Register handlers."""
        self.register_handlers(
            {
                "cursor_drag": self.handle_cursor_drag,
                "cursor_reached_destination": self.handle_cursor_reached_destination,
                "cursor_set_destination": self.handle_cursor_set_destination,
                "cursor_set_movement_mode": self.handle_cursor_set_movement_mode,
            },
        )

    async def handle_cursor_drag(self, event: Event[Iterable[int]]) -> None:
        """Drag one or more tiles."""
        await trio.lowlevel.checkpoint()
        for tile_color in event.data:
            if tile_color == Tile.one:
                self.tiles.insert(0, tile_color)
            else:
                self.tiles.append(tile_color)
        self.update_image()

    async def handle_cursor_reached_destination(
        self,
        event: Event[None],
    ) -> None:
        """Stop ticking."""
        self.unregister_handler_type("tick")
        await trio.lowlevel.checkpoint()

    def move_to_front(self) -> None:
        """Move this sprite to front."""
        group: sprite.LayeredDirty = self.groups()[-1]
        group.move_to_front(self)

    async def handle_cursor_set_destination(
        self,
        event: Event[tuple[int, int]],
    ) -> None:
        """Start moving towards new destination."""
        targeting: sprite.TargetingComponent = self.get_component("targeting")
        targeting.destination = event.data
        if not self.has_handler("tick"):
            self.register_handler(
                "tick",
                targeting.move_destination_time_ticks,
            )
        self.move_to_front()
        await trio.lowlevel.checkpoint()

    async def handle_pygame_mouse_motion(
        self,
        event: Event[sprite.PygameMouseMotion],
    ) -> None:
        """Set location to event data."""
        self.move_to_front()
        self.location = event.data["pos"]
        await trio.lowlevel.checkpoint()

    async def handle_cursor_set_movement_mode(
        self,
        event: Event[bool],
    ) -> None:
        """Change cursor movement mode. True if client mode, False if server mode."""
        client_mode = event.data
        if client_mode:
            self.register_handler(
                "PygameMouseMotion",
                self.handle_pygame_mouse_motion,
            )
        else:
            self.unregister_handler_type("PygameMouseMotion")
        await trio.lowlevel.checkpoint()

    def get_held_count(self) -> int:
        """Return the number of held tiles."""
        return len(self.tiles)

    def is_holding(self) -> bool:
        """Return True if the mouse is dragging something."""
        return len(self.tiles) > 0

    def get_held_info(self) -> tuple[int, ...]:
        """Return tuple of currently held tiles."""
        return tuple(reversed(self.tiles))

    def drop(
        self,
        number: int | None = None,
    ) -> tuple[int, ...]:
        """Pop and return tiles the Cursor is carrying.

        If number is None, pops all tiles, otherwise only pops given count.
        """
        if number is None:
            tiles_copy = self.get_held_info()
            self.tiles.clear()
            self.update_image()
            return tiles_copy
        tiles: list[int] = []
        for _ in range(number):
            tiles.append(self.tiles.pop())
        self.update_image()
        return tuple(tiles)


class Grid(TileRenderer):
    """Grid object, used for boards and parts of other objects."""

    __slots__ = ("data", "size")

    def __init__(
        self,
        name: str,
        size: tuple[int, int],
        tile_separation: int | None = None,
        background: tuple[int, int, int] | None = TILEDEFAULT,
    ) -> None:
        """Grid Objects require a size and game at least."""
        super().__init__(name, tile_separation, background)

        self.size = size

        self.data = array(
            [Tile.blank for i in range(int(self.size[0] * self.size[1]))],
            int8,
        ).reshape(self.size)

    def get_tile(self, xy: tuple[int, int]) -> int:
        """Return tile color at given index."""
        x, y = xy
        return int(self.data[y, x])

    def update_image(
        self,
        offset: tuple[int, int] | None = None,
        extra_space: tuple[int, int] | None = None,
    ) -> None:
        """Update self.image."""
        self.clear_image(self.size, extra_space)

        width, height = self.size

        for y in range(height):
            for x in range(width):
                pos = (x, y)
                self.blit_tile(self.get_tile(pos), pos, offset)

    def fake_tile_exists(self, xy: tuple[int, int]) -> bool:
        """Return if tile at given position is a fake tile."""
        return self.get_tile(xy) < 0

    def place_tile(self, xy: tuple[int, int], tile_color: int) -> bool:
        """Place tile at given position."""
        x, y = xy
        self.data[y, x] = tile_color
        self.update_image()

    def pop_tile(self, xy: tuple[int, int], replace: int = Tile.blank) -> int:
        """Return popped tile from given position in the grid."""
        tile_color = self.get_tile(xy)
        self.place_tile(xy, replace)
        return tile_color

    def get_colors(self) -> set[int]:
        """Return a set of the colors of tiles within self."""
        colors = set()
        width, height = self.size
        for y in range(height):
            for x in range(width):
                colors.add(self.get_tile((x, y)))
        return colors

    def is_empty(self, empty_color: int = Tile.blank) -> bool:
        """Return True if Grid is empty (all tiles are empty_color)."""
        colors = self.get_colors()
        return len(colors) == 1 and colors.pop() == empty_color


class Board(Grid):
    """Represents the board in the Game."""

    __slots__ = ("board_id",)

    def __init__(self, board_id: int) -> None:
        """Initialize player's board."""
        super().__init__(f"board_{board_id}", (5, 5), background=ORANGE)

        self.board_id = board_id

        self.update_location_on_resize = True

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.variant_play})"

    def bind_handlers(self) -> None:
        """Register event handlers."""
        self.register_handlers(
            {
                "game_board_data": self.handle_game_board_data,
            },
        )

    async def handle_game_board_data(
        self,
        event: Event[tuple[int, NDArray[int8]]],
    ) -> None:
        """Handle `game_board_data` event."""
        board_id, array = event.data

        if board_id != self.board_id:
            await trio.lowlevel.checkpoint()
            return

        self.data = array
        self.update_image()
        self.visible = True

        await trio.lowlevel.checkpoint()


class Row(TileRenderer):
    """Represents one of the five rows each player has."""

    __slots__ = ("color", "size", "tiles")
    greyshift = GREYSHIFT

    def __init__(
        self,
        name: str,
        size: int,
        tile_separation: int | None = None,
        background: tuple[int, int, int] | None = None,
    ) -> None:
        """Initialize row."""
        super().__init__(
            name,
            tile_separation,
            background,
        )

        self.color = Tile.blank
        self.size = int(size)
        self.count = 0

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.size})"

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image((self.size, 1))

        for x in range(self.count):
            self.blit_tile(self.color, (x, 0))
        for x in range(self.count, self.size):
            self.blit_tile(Tile.blank, (x, 0))

    def get_tile_point(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> int | None:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        pos = super().get_tile_point(screen_location)
        if pos is None:
            return None
        return pos[0]

    def get_placed(self) -> int:
        """Return the number of tiles in self that are not fake tiles, like grey ones."""
        return self.count

    def get_placeable(self) -> int:
        """Return the number of tiles permitted to be placed on self."""
        return self.size - self.get_placed()

    def is_full(self) -> bool:
        """Return True if this row is full."""
        return self.get_placeable() == 0

    def set_background(self, color: tuple[int, int, int] | None) -> None:
        """Set the background color for this row."""
        self.background = color
        self.update_image()


class FloorLine(Row):
    """Represents a player's floor line."""

    __slots__ = ("floor_line_id", "numbers", "text")

    def __init__(self, floor_line_id: int) -> None:
        """Initialize floor line."""
        super().__init__(f"floor_line_{floor_line_id}", 7, background=ORANGE)

        # self.font = Font(FONT, round(self.tile_size*1.2), color=BLACK, cx=False, cy=False)
        self.text = objects.Text(
            round(self.tile_size * 1.2),
            BLACK,
            cx=False,
            cy=False,
        )

        self.numbers = [-255 for _ in range(self.size)]

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.player!r})"

    def render(self, surface: pygame.surface.Surface) -> None:
        """Update self.image."""
        super().render(surface)

        sx, sy = self.location
        assert self.width_height is not None, "Should be impossible."
        w, h = self.width_height
        for x in range(self.size):
            xy = round(
                x * self.tile_full + self.tile_separation + sx - w / 2,
            ), round(
                self.tile_separation + sy - h / 2,
            )
            self.text.update_value(str(self.numbers[x]))
            self.text.location = Vector2(*xy)
            self.text.render(surface)


class Factory(TileRenderer):
    """Represents a Factory."""

    __slots__ = ("factory_id", "tiles")
    color = WHITE
    outline = BLUE

    def __init__(self, factory_id: int) -> None:
        """Initialize factory."""
        super().__init__(f"Factory_{factory_id}", background=None)

        self.factory_id = factory_id
        self.tiles: Counter[int] = Counter()

        self.update_location_on_resize = True

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.number})"

    def bind_handlers(self) -> None:
        """Register event handlers."""
        self.register_handlers(
            {
                "game_factory_data": self.handle_factory_data,
            },
        )

    async def handle_factory_data(
        self,
        event: Event[tuple[int, Counter[int]]],
    ) -> None:
        """Handle `game_factory_data` event."""
        factory_id, tiles = event.data

        if factory_id != self.factory_id:
            await trio.lowlevel.checkpoint()
            return

        self.tiles = tiles
        self.update_image()
        self.visible = True

        await trio.lowlevel.checkpoint()

    def update_image(self) -> None:
        """Update image."""
        self.clear_image((2, 2), extra=(16, 16))

        radius = 29
        pygame.draw.circle(
            self.image,
            self.outline,
            (radius, radius),
            radius,
        )
        pygame.draw.circle(
            self.image,
            self.color,
            (radius, radius),
            math.ceil(radius * 0.9),
        )

        for index, tile_color in enumerate(self.tiles.elements()):
            y, x = divmod(index, 2)
            self.blit_tile(tile_color, (x, y), (8, 8))

    def get_tile_point(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> tuple[int, int] | None:
        """Get tile point accounting for offset."""
        return super().get_tile_point(
            Vector2.from_iter(screen_location) - (8, 8),
        )


class TableCenter(TileRenderer):
    """sprite.Sprite that represents the center of the table."""

    __slots__ = ("tiles",)
    size = (6, 6)

    def __init__(self) -> None:
        """Initialize center of table."""
        super().__init__("TableCenter", background=None)

        self.tiles: Counter[int] = Counter()
        self.update_image()
        self.visible = True

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}()"

    def iter_tiles(self) -> Generator[int, None, None]:
        """Yield tile colors."""
        count = 0
        for tile_type in sorted(set(self.tiles) - {Tile.one}):
            tile_count = self.tiles[tile_type]
            for _ in range(tile_count):
                yield tile_type
                count += 1

        width, height = self.size
        remaining = width * height - count

        one_count = self.tiles.get(Tile.one, 0)
        remaining = max(remaining - one_count, 0)
        for _ in range(remaining):
            yield Tile.blank
        for _ in range(one_count):
            yield Tile.one

    def update_image(self) -> None:
        """Reset/update image."""
        self.clear_image(self.size)

        width, height = self.size
        tile_generator = self.iter_tiles()
        for y in range(height):
            for x in range(width):
                tile = next(tile_generator)
                # if tile == Tile.blank:
                #    continue
                self.blit_tile(tile, (x, y))
        self.dirty = 1

    def add_tile(self, tile: int) -> None:
        """Add a tile to the center of the table."""
        self.tiles.update((tile,))
        self.update_image()

    def add_tiles(self, tiles: Iterable[int]) -> None:
        """Add multiple int Objects to the Table Center Grid."""
        self.tiles.update(tiles)
        self.update_image()

    def pull_tiles(self, tile_color: int) -> list[int]:
        """Pop all of tile_color. Raises KeyError if not exists."""
        tile_count = self.tiles.pop(tile_color)
        return [tile_color] * tile_count


class HaltState(AsyncState["AzulClient"]):
    """Halt state to set state to None so running becomes False."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize Halt State."""
        super().__init__("Halt")

    async def check_conditions(self) -> None:
        """Set active state to None."""
        assert self.machine is not None
        await self.machine.set_state(None)


class GameState(AsyncState["AzulClient"]):
    """Checkers Game Asynchronous State base class."""

    __slots__ = ("id", "manager")

    def __init__(self, name: str) -> None:
        """Initialize Game State."""
        super().__init__(name)

        self.id: int = 0
        self.manager = ComponentManager(self.name)

    def add_actions(self) -> None:
        """Add internal component manager to state machine's component manager."""
        assert self.machine is not None
        self.machine.manager.add_component(self.manager)

    def group_add(self, new_sprite: sprite.Sprite) -> None:
        """Add new sprite to state machine's group."""
        assert self.machine is not None
        group = self.machine.get_group(self.id)
        assert group is not None, "Expected group from new group id"
        group.add(new_sprite)
        self.manager.add_component(new_sprite)

    async def exit_actions(self) -> None:
        """Remove group and unbind all components."""
        assert self.machine is not None
        self.machine.remove_group(self.id)
        self.manager.unbind_components()
        self.id = 0

    def change_state(
        self,
        new_state: str | None,
    ) -> Callable[[Event[Any]], Awaitable[None]]:
        """Return an async function that will change state to `new_state`."""

        async def set_state(*args: object, **kwargs: object) -> None:
            play_sound("button_click")
            await self.machine.set_state(new_state)

        return set_state


class KwargOutlineText(objects.OutlinedText):
    """Outlined objects.Text with attributes settable via keyword arguments."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        font: pygame.font.Font,
        **kwargs: object,
    ) -> None:
        """Initialize attributes via keyword arguments."""
        super().__init__(name, font)

        for key, value in kwargs.items():
            setattr(self, key, value)


class KwargButton(objects.Button):
    """objects.Button with attributes settable via keyword arguments."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        font: pygame.font.Font,
        **kwargs: object,
    ) -> None:
        """Initialize attributes via keyword arguments."""
        super().__init__(name, font)

        for key, value in kwargs.items():
            setattr(self, key, value)


class MenuState(GameState):
    """Game State where there is a menu with buttons."""

    button_minimum = 10
    fontsize = BUTTONFONTSIZE

    def __init__(self, name: str) -> None:
        """Initialize GameState and set up self.bh."""
        super().__init__(name)

    def add_button(
        self,
        name: str,
        value: str,
        action: Callable[[], None],
        location: tuple[int, int] | None = None,
        size: int = fontsize,
        minlen: int = button_minimum,
    ) -> int:
        """Add a new objects.Button object to group."""
        button = KwargButton(
            name,
            font=pygame.font.Font(FONT, size),
            visible=True,
            color=Color(0, 0, 0),
            text=value,
            location=location,
            handle_click=action,
        )
        self.group_add(button)

    def add_text(
        self,
        name: str,
        value: str,
        location: tuple[int, int],
        color: tuple[int, int, int] = BUTTON_TEXT_COLOR,
        size: int = fontsize,
        outline: tuple[int, int, int] = BUTTON_TEXT_OUTLINE,
    ) -> int:
        """Add a new objects.Text object to self.game with arguments. Return text id."""
        text = KwargOutlineText(
            name,
            font=pygame.font.Font(FONT, size),
            visible=True,
            color=color,
            text=value,
            location=location,
        )
        self.group_add(text)

    def set_var(self, attribute: str, value: object) -> None:
        """Set MenuState.{attribute} to {value}."""
        setattr(self, attribute, value)

    def to_state(self, new_state: str) -> Callable[[], Awaitable[None]]:
        """Return a function that will change game state to state_name."""

        async def set_state(*args: object, **kwargs: object) -> None:
            play_sound("button_click")
            await self.machine.set_state(new_state)

        return set_state

    def var_dependant_to_state(
        self,
        **kwargs: tuple[str, object],
    ) -> Callable[[], None]:
        """Attribute name = (target value, on trigger next_state)."""
        for state in kwargs:
            if not len(kwargs[state]) == 2:
                raise ValueError(f'Key "{state}" is invalid!')
            key, _value = kwargs[state]
            if not hasattr(self, key):
                raise ValueError(
                    f'{self} object does not have attribute "{key}"!',
                )

        def to_state_by_attributes() -> None:
            """Set MenuState.next_state to a new state if conditions are right."""
            for state in kwargs:
                key, value = kwargs[state]
                if getattr(self, key) == value:
                    self.next_state = state

        return to_state_by_attributes

    def with_update(
        self,
        update_function: Callable[[], None],
    ) -> Callable[[Callable[[], None]], Callable[[], None]]:
        """Return a wrapper for a function that will call update_function after function."""

        def update_wrapper(function: Callable[[], None]) -> Callable[[], None]:
            """Wrap anything that might require a screen update."""

            @wraps(function)
            def function_with_update() -> None:
                """Call main function, then update function."""
                function()
                update_function()

            return function_with_update

        return update_wrapper

    def update_text(
        self,
        text_name: str,
        value_function: Callable[[], str],
    ) -> Callable[[], None]:
        """Update text object with text_name's display value."""

        def updater() -> None:
            """Update text object {text_name}'s value with {value_function}."""
            assert self.game is not None
            text = self.game.get_object_by_name(f"objects.Text{text_name}")
            assert isinstance(text, objects.Text)
            text.update_value(value_function())

        return updater

    def toggle_button_state(
        self,
        textname: str,
        boolattr: str,
        textfunc: Callable[[bool], str],
    ) -> Callable[[], None]:
        """Return function that will toggle the value of text object <textname>, toggling attribute <boolattr>, and setting text value with textfunc."""

        def valfunc() -> str:
            """Return the new value for the text object. Gets called AFTER value is toggled."""
            return textfunc(getattr(self, boolattr))

        @self.with_update(self.update_text(textname, valfunc))
        def toggle_value() -> None:
            """Toggle the value of boolattr."""
            self.set_var(boolattr, not getattr(self, boolattr))

        return toggle_value


class InitializeState(AsyncState["AzulClient"]):
    """Initialize state."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize self."""
        super().__init__("initialize")

    async def check_conditions(self) -> str:
        """Go to title state."""
        return "title"


class TitleState(MenuState):
    """Game state when the title screen is up."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize title."""
        super().__init__("title")

    async def entry_actions(self) -> None:
        """Set up buttons."""
        assert self.machine is not None
        self.id = self.machine.new_group("title")

        button_font = pygame.font.Font(FONT, 28)
        title_font = pygame.font.Font(FONT, 56)

        title_text = KwargOutlineText(
            "title_text",
            title_font,
            visible=True,
            color=Color(0, 0, 0),
            outline=(255, 0, 0),
            border_width=4,
            text=__title__.upper(),
        )
        title_text.location = (SCREEN_SIZE[0] // 2, title_text.rect.h)
        self.group_add(title_text)

        hosting_button = KwargButton(
            "hosting_button",
            button_font,
            visible=True,
            color=Color(0, 0, 0),
            text="Host Networked Game",
            location=[x // 2 for x in SCREEN_SIZE],
            handle_click=self.change_state("play_hosting"),
        )
        self.group_add(hosting_button)

        join_button = KwargButton(
            "join_button",
            button_font,
            visible=True,
            color=Color(0, 0, 0),
            text="Join Networked Game",
            location=hosting_button.location
            + Vector2(
                0,
                hosting_button.rect.h + 10,
            ),
            handle_click=self.change_state("play_joining"),
        )
        self.group_add(join_button)

        internal_button = KwargButton(
            "internal_hosting",
            button_font,
            visible=True,
            color=Color(0, 0, 0),
            text="Singleplayer Game",
            location=hosting_button.location
            - Vector2(
                0,
                hosting_button.rect.h + 10,
            ),
            handle_click=self.change_state("play_hosting_internal"),
        )
        self.group_add(internal_button)

        quit_button = KwargButton(
            "quit_button",
            button_font,
            visible=True,
            color=Color(0, 0, 0),
            text="Quit",
            location=join_button.location
            + Vector2(
                0,
                join_button.rect.h + 10,
            ),
            handle_click=self.change_state("Halt"),
        )
        self.group_add(quit_button)


class CreditsState(MenuState):
    """Game state when credits for original game are up."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize credits."""
        super().__init__("credits")

    def check_state(self) -> str:
        """Return to title."""
        return "title"


class SettingsState(MenuState):
    """Game state when user is defining game type, players, etc."""

    def __init__(self) -> None:
        """Initialize settings."""
        super().__init__("settings")

        self.player_count = 0  # 2
        self.host_mode = True
        self.variant_play = False

    async def entry_actions(self) -> None:
        """Add cursor object and tons of button and text objects to the game."""
        await super().entry_actions()

        def add_numbers(
            start: int,
            end: int,
            width_each: int,
            cx: int,
            cy: int,
        ) -> None:
            """Add numbers."""
            count = end - start + 1
            evencount = count % 2 == 0
            mid = count // 2

            def add_number(
                number: int,
                display: str | int,
            ) -> None:
                """Add number."""
                if evencount:
                    if number < mid:
                        x = number - start - 0.5
                    else:
                        x = number - mid + 0.5
                else:
                    if number < mid:
                        x = number - start + 1
                    elif number == mid:
                        x = 0
                    else:
                        x = number - mid

                @self.with_update(
                    self.update_text(
                        "Players",
                        lambda: f"Players: {self.player_count}",
                    ),
                )
                def set_player_count() -> None:
                    """Set variable player_count to {display} while updating text."""
                    return self.set_var("player_count", display)

                self.add_button(
                    f"SetCount{number}",
                    str(display),
                    set_player_count,
                    (int(cx + (width_each * x)), int(cy)),
                    size=int(self.fontsize / 1.5),
                    minlen=3,
                )

            for i in range(count):
                add_number(i, start + i)

        sw, sh = SCREEN_SIZE
        cx = sw // 2
        cy = sh // 2

        def host_text(x: object) -> str:
            return f"Host Mode: {x}"

        self.add_text(
            "Host",
            host_text(self.host_mode),
            (cx, cy - self.bh * 3),
        )
        self.add_button(
            "ToggleHost",
            "Toggle",
            self.toggle_button_state("Host", "host_mode", host_text),
            (cx, cy - self.bh * 2),
            size=int(self.fontsize / 1.5),
        )

        ##        # TEMPORARY: Hide everything to do with "Host Mode", networked games aren't done yet.
        ##        assert self.game is not None
        ##        self.game.set_attr_all("visible", False)

        def varient_text(x: object) -> str:
            return f"Variant Play: {x}"

        self.add_text(
            "Variant",
            varient_text(self.variant_play),
            (cx, cy - self.bh),
        )
        self.add_button(
            "ToggleVarient",
            "Toggle",
            self.toggle_button_state("Variant", "variant_play", varient_text),
            (cx, cy),
            size=int(self.fontsize / 1.5),
        )

        self.add_text(
            "Players",
            f"Players: {self.player_count}",
            (cx, cy + self.bh),
        )
        add_numbers(2, 4, 70, cx, int(cy + self.bh * 2))

        var_to_state = self.var_dependant_to_state(
            FactoryOffer=("host_mode", True),
            FactoryOfferNetworked=("host_mode", False),
        )
        self.add_button(
            "StartGame",
            "Start Game",
            var_to_state,
            (cx, cy + self.bh * 3),
        )

    def exit_actions(self) -> None:
        """Start game."""
        assert self.game is not None
        self.game.start_game(
            self.player_count,
            self.variant_play,
            self.host_mode,
        )
        self.game.bag.full_reset()


class PhaseFactoryOffer(GameState):
    """Game state when it's the Factory Offer Stage."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize factory offer phase."""
        super().__init__("FactoryOffer")


class PhaseFactoryOfferNetworked(PhaseFactoryOffer):
    """Factory offer phase but networked."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize factory offer networked."""
        GameState.__init__(self, "FactoryOfferNetworked")

    def check_state(self) -> str:
        """Go to networked wall tiling."""
        return "WallTilingNetworked"


class PhaseWallTiling(GameState):
    """Wall tiling game phase."""

    # __slots__ = ()
    def __init__(self) -> None:
        """Initialize will tiling phase."""
        super().__init__("WallTiling")

    def entry_actions(self) -> None:
        """Start wall tiling."""
        assert self.game is not None
        self.next_starter: int = 0
        self.not_processed = []

        self.game.player_turn_over()

        # For each player,
        for player_id in range(self.game.players):
            # Activate wall tiling mode.
            player = self.game.get_player(player_id)
            player.wall_tiling()
            # Add that player's player_id to the list of not-processed players.
            self.not_processed.append(player.player_id)

        # Start processing players.
        self.game.next_turn()

    def do_actions(self) -> None:
        """Do game actions."""
        assert self.game is not None
        if self.not_processed:
            if self.game.player_turn in self.not_processed:
                player = self.game.get_player(self.game.player_turn)
                if player.done_wall_tiling():
                    # Once player is done wall tiling, score their moves.
                    # Also gets if they had the number one tile.
                    number_one = player.score_phase()

                    if number_one:
                        # If player had the number one tile, remember that.
                        self.next_starter = self.game.player_turn
                        # Then, add the number one tile back to the table center.
                        table = self.game.get_object_by_name("TableCenter")
                        assert isinstance(table, TableCenter)
                        table.add_number_one_tile()
                    # After calculating their score, delete player from un-processed list
                    self.not_processed.remove(self.game.player_turn)
                    # and continue to the next un-processed player.
                    self.game.next_turn()
            else:
                self.game.next_turn()

    def check_state(self) -> str | None:
        """Go to next state if ready."""
        assert self.game is not None
        cursor = self.game.get_object_by_name("Cursor")
        assert isinstance(cursor, Cursor)
        if not self.not_processed and not cursor.is_holding():
            return "PrepareNext"
        return None

    def exit_actions(self) -> None:
        """Update who's turn it is."""
        assert self.game is not None
        # Set up the player that had the number one tile to be the starting player next round.
        self.game.player_turn_over()
        # Goal: make (self.player_turn + 1) % self.players = self.next_starter
        nturn = self.next_starter - 1
        if nturn < 0:
            nturn += self.game.players
        self.game.player_turn = nturn


class PhasePrepareNext(GameState):
    """Prepare next phase of game."""

    __slots__ = ("new_round",)

    def __init__(self) -> None:
        """Initialize prepare next state."""
        super().__init__("PrepareNext")
        self.new_round = False

    def entry_actions(self) -> None:
        """Find out if game continues."""
        assert self.game is not None
        players = (
            self.game.get_player(player_id)
            for player_id in range(self.game.players)
        )
        complete = (player.has_horzontal_line() for player in players)
        self.new_round = not any(complete)

    def check_state(self) -> str:
        """Go to factory offer if new round else end screen."""
        if self.new_round:
            return "FactoryOffer"
        return "End"


class EndScreen(MenuState):
    """End screen state."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize end screen."""
        super().__init__("End")


class PlayHostingState(AsyncState["AzulClient"]):
    """Start running server."""

    __slots__ = ("address",)

    internal_server = False

    def __init__(self) -> None:
        """Initialize Play internal hosting / hosting State."""
        extra = "_internal" if self.internal_server else ""
        super().__init__(f"play_hosting{extra}")

    async def entry_actions(self) -> None:
        """Start hosting server."""
        assert self.machine is not None
        self.machine.manager.add_components(
            (
                GameServer(self.internal_server),
                GameClient("network"),
            ),
        )

        host = "localhost" if self.internal_server else await find_ip()
        port = DEFAULT_PORT

        self.address = (host, port)

        await self.machine.raise_event(Event("server_start", self.address))

    async def exit_actions(self) -> None:
        """Have client connect."""
        assert self.machine is not None
        await self.machine.raise_event(
            Event("client_connect", self.address),
        )

    async def check_conditions(self) -> str | None:
        """Return to Play state when server is up and running."""
        server: GameServer = self.machine.manager.get_component("GameServer")
        return "play" if server.running else None


class PlayInternalHostingState(PlayHostingState):
    """Host server with internal server mode."""

    __slots__ = ()

    internal_server = True


class ReturnElement(element_list.Element, objects.Button):
    """Connection list return to title element sprite."""

    __slots__ = ()

    def __init__(self, name: str, font: pygame.font.Font) -> None:
        """Initialize return element."""
        super().__init__(name, font)

        self.update_location_on_resize = False
        self.border_width = 4
        self.outline = RED
        self.text = "Return to Title"
        self.visible = True
        self.location = (SCREEN_SIZE[0] // 2, self.location.y + 10)

    async def handle_click(
        self,
        _: Event[sprite.PygameMouseButtonEventData],
    ) -> None:
        """Handle Click Event."""
        await self.raise_event(
            Event("return_to_title", None, 2),
        )


class ConnectionElement(element_list.Element, objects.Button):
    """Connection list element sprite."""

    __slots__ = ()

    def __init__(
        self,
        name: tuple[str, int],
        font: pygame.font.Font,
        motd: str,
    ) -> None:
        """Initialize connection element."""
        super().__init__(name, font)

        self.text = f"[{name[0]}:{name[1]}]\n{motd}"
        self.visible = True

    async def handle_click(
        self,
        _: Event[sprite.PygameMouseButtonEventData],
    ) -> None:
        """Handle Click Event."""
        details = self.name
        await self.raise_event(
            Event("join_server", details, 2),
        )


class PlayJoiningState(GameState):
    """Start running client."""

    __slots__ = ("font",)

    def __init__(self) -> None:
        """Initialize Joining State."""
        super().__init__("play_joining")

        self.font = pygame.font.Font(
            FONT,
            12,
        )

    async def entry_actions(self) -> None:
        """Add game client component."""
        await super().entry_actions()
        assert self.machine is not None
        self.id = self.machine.new_group("join")
        client = GameClient("network")

        # Add network to higher level manager
        self.machine.manager.add_component(client)

        connections = element_list.ElementList("connection_list")
        self.manager.add_component(connections)
        group = self.machine.get_group(self.id)
        assert group is not None
        group.add(connections)

        return_font = pygame.font.Font(
            FONT,
            30,
        )
        return_button = ReturnElement("return_button", return_font)
        connections.add_element(return_button)

        self.manager.register_handlers(
            {
                "update_listing": self.handle_update_listing,
                "return_to_title": self.handle_return_to_title,
                "join_server": self.handle_join_server,
            },
        )

        await self.manager.raise_event(Event("update_listing", None))

    async def handle_update_listing(self, _: Event[None]) -> None:
        """Update server listing."""
        assert self.machine is not None

        connections = self.manager.get_component("connection_list")

        old: list[tuple[str, int]] = []
        current: list[tuple[str, int]] = []

        # print(f'{self.machine.active_state = }')
        # print(f'{self.name = }')
        while (
            self.machine.active_state is not None
            and self.machine.active_state is self
        ):
            # print("handle_update_listing click")

            for motd, details in await read_advertisements():
                current.append(details)
                if connections.component_exists(details):
                    continue
                element = ConnectionElement(details, self.font, motd)
                element.rect.topleft = (
                    connections.get_new_connection_position()
                )
                element.rect.topleft = (10, element.location.y + 3)
                connections.add_element(element)
            for details in old:
                if details in current:
                    continue
                connections.delete_element(details)
            old, current = current, []

    async def handle_join_server(self, event: Event[tuple[str, int]]) -> None:
        """Handle join server event."""
        details = event.data
        await self.machine.raise_event(
            Event("client_connect", details),
        )
        await self.machine.set_state("play")

    async def handle_return_to_title(self, _: Event[None]) -> None:
        """Handle return to title event."""
        # Fire server stop event so server shuts down if it exists
        await self.machine.raise_event_internal(Event("network_stop", None))

        if self.machine.manager.component_exists("network"):
            self.machine.manager.remove_component("network")

        await self.machine.set_state("title")


# async def check_conditions(self) -> str | None:
# return None


##    async def entry_actions(self) -> None:
##        """Set up buttons."""
##        assert self.machine is not None
##        self.id = self.machine.new_group("initialize")
##
##        self.group_add(Cursor())
##        await self.manager.raise_event(Event("cursor_drag", [3, 5]))
##        self.manager.register_handler("PygameMouseMotion", self.mouse_moved)
##
##        ##        board = Board()
##        ####        board.place_tile((2, 2), Tile.red)
##        ##        board.location = Vector2.from_iter(SCREEN_SIZE) // 2
##        ##        self.group_add(board)
##
##        center.add_tiles((0, 1, 2, 3, 5))


class PlayState(GameState):
    """Game Play State."""

    __slots__ = ("current_turn", "exit_data")

    def __init__(self) -> None:
        """Initialize Play State."""
        super().__init__("play")

        self.current_turn: int = 0

        # (0: normal | 1: error) <error message> <? handled>
        self.exit_data: tuple[int, str, bool] | None = None

    def register_handlers(self) -> None:
        """Register event handlers."""
        self.manager.register_handlers(
            {
                "game_initial_config": self.handle_game_initial_config,
                "client_disconnected": self.handle_client_disconnected,
                "game_winner": self.handle_game_over,
            },
        )

    def add_actions(self) -> None:
        """Register handlers."""
        super().add_actions()
        self.register_handlers()

    async def entry_actions(self) -> None:
        """Add GameBoard and raise init event."""
        assert self.machine is not None
        if self.id == 0:
            self.id = self.machine.new_group("play")

        self.group_add(Cursor())

        center = TableCenter()
        center.location = Vector2.from_iter(SCREEN_SIZE) // 2
        self.group_add(center)

        # self.group_add(())
        ##        gameboard = GameBoard(
        ##            45,
        ##        )
        ##        gameboard.location = [x // 2 for x in SCREEN_SIZE]
        ##        self.group_add(gameboard)

    async def handle_game_initial_config(
        self,
        event: Event[tuple[bool, int, int, int]],
    ) -> None:
        """Handle `game_initial_config` event."""
        varient_play, player_count, factory_count, self.current_turn = (
            event.data
        )

        center = Vector2.from_iter(SCREEN_SIZE) // 2

        # Add factories
        for index, degrees in enumerate(range(0, 360, 360 // factory_count)):
            factory = Factory(index)
            factory.location = (
                Vector2.from_degrees(
                    degrees - 90,
                    145,
                )
                + center
            )
            self.group_add(factory)

        # Add players
        for index, degrees in enumerate(range(0, 360, 360 // player_count)):
            board = Board(index)
            board.location = (
                Vector2.from_degrees(
                    degrees - 45,
                    300,
                )
                + center
            )
            self.group_add(board)

    async def check_conditions(self) -> str | None:
        """Return to title if client component doesn't exist."""
        if not self.machine.manager.component_exists("network"):
            return "title"
        return None

    async def exit_actions(self) -> None:
        """Raise network stop event and remove components."""
        # Fire server stop event so server shuts down if it exists
        # await self.machine.raise_event(Event("network_stop", None))
        await self.machine.raise_event_internal(Event("network_stop", None))

        if self.machine.manager.component_exists("network"):
            self.machine.manager.remove_component("network")
        if self.machine.manager.component_exists("GameServer"):
            self.machine.manager.remove_component("GameServer")

        # Unbind components and remove group
        await super().exit_actions()

        self.register_handlers()

        assert self.manager.has_handler("game_winner")

    async def handle_game_over(self, event: Event[int]) -> None:
        """Handle game over event."""
        winner = event.data
        self.exit_data = (0, f"{winner} Won", False)

        await self.machine.raise_event_internal(Event("network_stop", None))

    async def handle_client_disconnected(self, event: Event[str]) -> None:
        """Handle client disconnected error."""
        error = event.data
        print(f"handle_client_disconnected  {error = }")

        self.exit_data = (1, f"Client Disconnected$${error}", False)

    # await self.do_actions()

    async def do_actions(self) -> None:
        """Perform actions for this State."""
        # print(f"{self.__class__.__name__} do_actions tick")
        if self.exit_data is None:
            return

        exit_status, message, handled = self.exit_data

        if handled:
            return
        self.exit_data = (exit_status, message, True)

        font = pygame.font.Font(
            FONT,
            28,
        )

        error_message = ""
        if exit_status == 1:
            message, error_message = message.split("$$")

        if not self.manager.component_exists("continue_button"):
            continue_button = KwargButton(
                "continue_button",
                font,
                visible=True,
                color=Color(0, 0, 0),
                text=f"{message} - Return to Title",
                location=[x // 2 for x in SCREEN_SIZE],
                handle_click=self.change_state("title"),
            )
            self.group_add(continue_button)
            group = continue_button.groups()[0]
            # LayeredDirty, not just AbstractGroup
            group.move_to_front(continue_button)  # type: ignore[attr-defined]
        else:
            continue_button = self.manager.get_component("continue_button")

        if exit_status == 1:
            if not self.manager.component_exists("error_text"):
                error_text = objects.OutlinedText("error_text", font)
            else:
                error_text = self.manager.get_component("error_text")
            error_text.visible = True
            error_text.color = Color(255, 0, 0)
            error_text.border_width = 1
            error_text.text += error_message + "\n"
            error_text.location = continue_button.location + Vector2(
                0,
                continue_button.rect.h + 10,
            )

            if not self.manager.component_exists("error_text"):
                self.group_add(error_text)


class AzulClient(sprite.GroupProcessor):
    """Azul Game Client."""

    __slots__ = ("manager",)

    def __init__(self, manager: ExternalRaiseManager) -> None:
        """Initialize Checkers Client."""
        super().__init__()
        self.manager = manager

        self.add_states(
            (
                HaltState(),
                InitializeState(),
                TitleState(),
                CreditsState(),
                SettingsState(),
                PlayHostingState(),
                PlayInternalHostingState(),
                PlayJoiningState(),
                PlayState(),
            ),
        )

    async def raise_event(self, event: Event[Any]) -> None:
        """Raise component event in all groups."""
        await self.manager.raise_event(event)

    async def raise_event_internal(self, event: Event[Any]) -> None:
        """Raise component event in all groups."""
        await self.manager.raise_event_internal(event)


async def async_run() -> None:
    """Run program."""
    # Set up globals
    global SCREEN_SIZE

    # Set up the screen
    screen = pygame.display.set_mode(SCREEN_SIZE, RESIZABLE, 16, vsync=VSYNC)
    pygame.display.set_caption(f"{__title__} v{__version__}")
    # pygame.display.set_icon(pygame.image.load('icon.png'))
    pygame.display.set_icon(get_tile_image(Tile.one, 32))
    screen.fill((0xFF, 0xFF, 0xFF))

    ##    try:
    async with trio.open_nursery() as main_nursery:
        event_manager = ExternalRaiseManager(
            "checkers",
            main_nursery,  # "client"
        )
        client = AzulClient(event_manager)

        background = pygame.image.load(
            DATA_FOLDER / "background.png",
        ).convert()
        client.clear(screen, background)

        client.set_timing_threshold(1000 / 80)

        await client.set_state("initialize")

        music_end = USEREVENT + 1  # This event is sent when a music track ends

        # Set music end event to our new event
        pygame.mixer.music.set_endevent(music_end)

        # Load and start playing the music
        # pygame.mixer.music.load('sound/')
        # pygame.mixer.music.play()

        clock = Clock()

        resized_window = False
        while client.running:
            async with trio.open_nursery() as event_nursery:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        await client.set_state("Halt")
                    elif event.type == KEYUP and event.key == K_ESCAPE:
                        pygame.event.post(pygame.event.Event(QUIT))
                    elif event.type == music_end:
                        # If the music ends, stop it and play it again.
                        pygame.mixer.music.stop()
                        pygame.mixer.music.play()
                    elif event.type == WINDOWRESIZED:
                        SCREEN_SIZE = (event.x, event.y)
                        resized_window = True
                    sprite_event = sprite.convert_pygame_event(event)
                    # print(sprite_event)
                    event_nursery.start_soon(
                        event_manager.raise_event,
                        sprite_event,
                    )
                event_nursery.start_soon(client.think)
                event_nursery.start_soon(clock.tick, FPS)

            await client.raise_event(
                Event(
                    "tick",
                    sprite.TickEventData(
                        time_passed=clock.get_time()
                        / 1e9,  # nanoseconds -> seconds
                        fps=clock.get_fps(),
                    ),
                ),
            )

            if resized_window:
                resized_window = False
                screen.fill((0xFF, 0xFF, 0xFF))
                rects = [Rect((0, 0), SCREEN_SIZE)]
                client.repaint_rect(rects[0])
                rects.extend(client.draw(screen))
            else:
                rects = client.draw(screen)
            pygame.display.update(rects)
    client.clear_groups()

    # Once the game has ended, stop the music
    pygame.mixer.music.stop()


def run() -> None:
    """Start asynchronous run."""
    trio.run(async_run)


def screenshot_last_frame() -> None:
    """Save the last frame before the game crashed."""
    surface = pygame.display.get_surface().copy()
    str_time = "_".join(time.asctime().split(" "))
    filename = f"Crash_at_{str_time}.png"

    path = Path("screenshots").absolute()
    if not path.exists():
        os.mkdir(path)

    fullpath = path / filename

    pygame.image.save(surface, fullpath, filename)
    del surface

    print(f'Saved screenshot to "{fullpath}".')


def cli_run() -> None:
    """Run from command line interface."""
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.\n")

    # Make sure the game will display correctly on high DPI monitors on Windows.
    if sys.platform == "win32":
        from ctypes import windll

        with contextlib.suppress(AttributeError):
            windll.user32.SetProcessDPIAware()
        del windll

    exception: str | None = None
    try:
        # Initialize Pygame
        _success, fail = pygame.init()
        if fail > 0:
            print(
                "Warning! Some modules of Pygame have not initialized properly!\n",
                "This can occur when not all required modules of SDL are installed.",
            )
        run()
    except ExceptionGroup as exc:
        print(exc)
        exception = traceback.format_exception(exc)
    ##        raise
    ##    except BaseException as ex:
    ##        screenshot_last_frame()
    ##        # errorbox.errorbox('Error', f'A {type(ex).__name__} Error Has Occored: {", ".join(ex.args)}')
    ##        raise
    finally:
        pygame.quit()
        if exception is not None:
            print("".join(exception), file=sys.stderr)


if __name__ == "__main__":
    cli_run()
