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
__version__ = "2.0.0"

import contextlib
import importlib
import math
import operator
import os
import random
import sys
import time
import traceback
from collections import Counter
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeVar

import pygame
import trio
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
from azul.component import (
    ComponentManager,
    Event,
    ExternalRaiseManager,
)
from azul.network_shared import DEFAULT_PORT, find_ip
from azul.server import GameServer
from azul.sound import SoundData, play_sound as base_play_sound
from azul.statemachine import AsyncState
from azul.tools import (
    lerp_color,
    saturate,
)
from azul.vector import Vector2

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
        Generator,
        Iterable,
        Sequence,
    )

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


from azul.state import Tile

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


class ObjectHandler:
    """ObjectHandler class, meant to be used for other classes."""

    # __slots__ = ("objects", "next_id", "cache")

    def __init__(self) -> None:
        """Initialize object handler."""
        self.objects: dict[int, Object] = {}
        self.next_id = 0
        self.cache: dict[str, int] = {}

        self.recalculate_render = True
        self._render_order: tuple[int, ...] = ()

    def add_object(self, obj: Object) -> None:
        """Add an object to the game."""
        obj.id = self.next_id
        self.objects[self.next_id] = obj
        self.next_id += 1
        self.recalculate_render = True

    def rm_object(self, obj: Object) -> None:
        """Remove an object from the game."""
        del self.objects[obj.id]
        self.recalculate_render = True

    def rm_star(self) -> None:
        """Remove all objects from self.objects."""
        for oid in list(self.objects):
            self.rm_object(self.objects[oid])
        self.next_id = 0

    def get_object(self, object_id: int) -> Object | None:
        """Return the object associated with object id given. Return None if object not found."""
        if object_id in self.objects:
            return self.objects[object_id]
        return None

    def get_objects_with_attr(self, attribute: str) -> tuple[int, ...]:
        """Return a tuple of object ids with given attribute."""
        return tuple(
            oid
            for oid in self.objects
            if hasattr(self.objects[oid], attribute)
        )

    def get_object_by_attr(
        self,
        attribute: str,
        value: object,
    ) -> tuple[int, ...]:
        """Return a tuple of object ids with <attribute> that are equal to <value>."""
        matches = []
        for oid in self.get_objects_with_attr(attribute):
            if getattr(self.objects[oid], attribute) == value:
                matches.append(oid)
        return tuple(matches)

    def get_object_given_name(self, name: str) -> tuple[int, ...]:
        """Return a tuple of object ids with names matching <name>."""
        return self.get_object_by_attr("name", name)

    def reset_cache(self) -> None:
        """Reset the cache."""
        self.cache = {}

    def get_object_by_name(self, object_name: str) -> Object:
        """Get object by name, with cache."""
        if object_name not in self.cache:
            ids = self.get_object_given_name(object_name)
            if ids:
                self.cache[object_name] = min(ids)
            else:
                raise RuntimeError(f"{object_name} Object Not Found!")
        result = self.get_object(self.cache[object_name])
        if result is None:
            raise RuntimeError(f"{object_name} Object Not Found!")
        return result

    def set_attr_all(self, attribute: str, value: object) -> None:
        """Set given attribute in all of self.objects to given value in all objects with that attribute."""
        for oid in self.get_objects_with_attr(attribute):
            setattr(self.objects[oid], attribute, value)

    def recalculate_render_order(self) -> None:
        """Recalculate the order in which to render objects to the screen."""
        new: dict[int, int] = {}
        cur = 0
        for oid in reversed(self.objects):
            obj = self.objects[oid]
            if hasattr(obj, "Render_Priority"):
                prior = obj.Render_Priority
                if isinstance(prior, str):
                    add = 0
                    if prior[:4] == "last":
                        try:
                            add = int(prior[4:] or 0)
                        except ValueError:
                            add = 0
                        pos = len(self.objects) + add
                    if prior[:5] == "first":
                        try:
                            add = int(prior[5:] or 0)
                        except ValueError:
                            add = 0
                        pos = -1 + add
                    if pos not in new.values():
                        new[oid] = pos
                    else:
                        while True:
                            if add < 0:
                                pos -= 1
                            else:
                                pos += 1
                            if pos not in new.values():
                                new[oid] = pos
                                break
                else:
                    try:
                        prior = int(prior)
                    except ValueError:
                        prior = cur
                    while True:
                        if prior in new.values():
                            prior += 1
                        else:
                            break
                    new[oid] = prior
            else:
                while True:
                    if cur in new.values():
                        cur += 1
                    else:
                        break
                new[oid] = cur
                cur += 1
        revnew = {new[k]: k for k in new}
        self._render_order = tuple(revnew[key] for key in sorted(revnew))

    def process_objects(self, time_passed: float) -> None:
        """Call the process function on all objects."""
        if self.recalculate_render:
            self.recalculate_render_order()
            self.recalculate_render = False
        for oid in iter(self.objects):
            self.objects[oid].process(time_passed)

    def render_objects(self, surface: pygame.surface.Surface) -> None:
        """Render all objects to surface."""
        if not self._render_order or self.recalculate_render:
            self.recalculate_render_order()
            self.recalculate_render = False
        for oid in self._render_order:  # reversed(list(self.objects.keys())):
            self.objects[oid].render(surface)

    def __del__(self) -> None:
        """Cleanup."""
        self.reset_cache()
        self.rm_star()


class MultipartObject(ObjectHandler):
    """Thing that is both an Object and an ObjectHandler, and is meant to be an Object made up of multiple Objects."""

    def __init__(self, name: str):
        """Initialize Object and ObjectHandler of self.

        Also set self._lastloc and self._lasthidden to None
        """
        ObjectHandler.__init__(self)

        self._lastloc: Vector2 | None = None
        self._lasthidden: bool | None = None

    def reset_position(self) -> None:
        """Reset the position of all objects within."""
        raise NotImplementedError

    def get_intersection(
        self,
        point: tuple[int, int] | Vector2,
    ) -> tuple[str, tuple[int, int]] | tuple[None, None]:
        """Return where a given point touches in self. Returns (None, None) with no intersections."""
        for oid in self.objects:
            obj = self.objects[oid]
            if hasattr(obj, "get_tile_point"):
                output = obj.get_tile_point(point)
                if output is not None:
                    return obj.name, output
            else:
                raise Warning(
                    "Not all of self.objects have the get_tile_point attribute!",
                )
        return None, None

    def process(self, time_passed: float) -> None:
        """Process Object self and ObjectHandler self and call self.reset_position on location change."""
        Object.process(self, time_passed)
        ObjectHandler.process_objects(self, time_passed)

        if self.location != self._lastloc:
            self.reset_position()
            self._lastloc = self.location

        if self.hidden != self._lasthidden:
            self.set_attr_all("hidden", self.hidden)
            self._lasthidden = self.hidden

    def render(self, surface: pygame.surface.Surface) -> None:
        """Render self and all parts to the surface."""
        Object.render(self, surface)
        ObjectHandler.render_objects(self, surface)

    def __del__(self) -> None:
        """Delete data."""
        Object.__del__(self)
        ObjectHandler.__del__(self)


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

    def clear_image(self, tile_dimensions: tuple[int, int]) -> None:
        """Reset self.image using tile_dimensions tuple and fills with self.background. Also updates self.width_height."""
        tile_width, tile_height = tile_dimensions
        tile_full = self.tile_size + self.tile_separation
        self.image = get_tile_container_image(
            (
                round(tile_width * tile_full + self.tile_separation),
                round(tile_height * tile_full + self.tile_separation),
            ),
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


##    def screen_size_update(self) -> None:
##        """Handle screensize is changes."""
##        nx, ny = self.location
##
##        if self.location_mode_on_resize == "Scale":
##            ow, oh = self.screen_size_last
##            nw, nh = SCREEN_SIZE
##
##            x, y = self.location
##            nx, ny = x * (nw / ow), y * (nh / oh)
##
##        self.location = Vector2(nx, ny)
##        self.screen_size_last = SCREEN_SIZE


class Cursor(TileRenderer):
    """Cursor TileRenderer.

    Registers following event handlers:
    - cursor_drag
    - cursor_reached_destination
    - cursor_set_destination
    - cursor_set_location
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

    def bind_handlers(self):
        """Register handlers."""
        self.register_handlers(
            {
                "cursor_drag": self.handle_cursor_drag,
                "cursor_reached_destination": self.handle_cursor_reached_destination,
                "cursor_set_destination": self.handle_cursor_set_destination,
                "cursor_set_location": self.handle_cursor_set_location,
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

    async def handle_cursor_set_location(
        self,
        event: Event[tuple[int, int]],
    ) -> None:
        """Set location to event data."""
        self.move_to_front()
        self.location = event.data
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

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image(self.size)

        width, height = self.size

        for y in range(height):
            for x in range(width):
                pos = (x, y)
                self.blit_tile(self.get_tile(pos), pos)

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

    __slots__ = ("additions", "variant_play", "wall_tiling")

    def __init__(self, variant_play: bool = False) -> None:
        """Initialize player's board."""
        super().__init__("Board", (5, 5), background=ORANGE)

        self.variant_play = variant_play
        self.additions: dict[int, int | None] = {}

        self.wall_tiling = False

        if not variant_play:
            self.set_colors()
        else:
            self.update_image()
        self.visible = True

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.variant_play})"

    def set_colors(self, keep_real: bool = True) -> None:
        """Reset tile colors."""
        width, height = self.size
        for y in range(height):
            for x in range(width):
                if not keep_real or self.fake_tile_exists((x, y)):
                    color = -((height - y + x) % REGTILECOUNT + 1)
                    self.data[y, x] = color
        self.update_image()

    def get_row(self, index: int) -> Generator[int, None, None]:
        """Return a row from self. Does not delete data from internal grid."""
        for x in range(self.size[0]):
            yield self.get_info((x, index))

    def get_column(self, index: int) -> Generator[int, None, None]:
        """Return a column from self. Does not delete data from internal grid."""
        for y in range(self.size[1]):
            yield self.get_info((index, y))

    def get_colors_in_row(
        self,
        index: int,
        exclude_negatives: bool = True,
    ) -> set[int]:
        """Return the colors placed in a given row in internal grid."""
        row_colors: Iterable[int] = self.get_row(index)
        if exclude_negatives:
            row_colors = (c for c in row_colors if c >= 0)
        return set(row_colors)

    def get_colors_in_column(
        self,
        index: int,
        exclude_negatives: bool = True,
    ) -> set[int]:
        """Return the colors placed in a given row in internal grid."""
        column_colors: Iterable[int] = self.get_column(index)
        if exclude_negatives:
            column_colors = (c for c in column_colors if c >= 0)
        return set(column_colors)

    def is_wall_tiling(self) -> bool:
        """Return True if in Wall Tiling Mode."""
        return self.wall_tiling

    def can_place_tile_color_at_point(
        self,
        position: tuple[int, int],
        tile_color: int,
    ) -> bool:
        """Return True if tile's color is valid at given position."""
        column, row = position
        colors = self.get_colors_in_column(column) | self.get_colors_in_row(
            row,
        )
        return tile_color not in colors

    ##    def remove_invalid_additions(self) -> None:
    ##        """Remove invalid additions that would not be placeable."""
    ##        # In the wall-tiling phase, it may happen that you
    ##        # are not able to move the rightmost tile of a certain
    ##        # pattern line over to the wall because there is no valid
    ##        # space left for it. In this case, you must immediately
    ##        # place all tiles of that pattern line in your floor line.
    ##        for row in range(self.size[1]):
    ##            row_tile = self.additions[row]
    ##            if not isinstance(row_tile, int):
    ##                continue
    ##            valid = self.calculate_valid_locations_for_tile_row(row)
    ##            if not valid:
    ##                floor = self.player.get_object_by_name("floor_line")
    ##                assert isinstance(floor, FloorLine)
    ##                floor.place_tile(row_tile)
    ##                self.additions[row] = None

    ##    def wall_tile_from_point(self, position: tuple[int, int]) -> bool:
    ##        """Given a position, wall tile. Return success on placement. Also updates if in wall tiling mode."""
    ##        success = False
    ##        column, row = position
    ##        at_point = self.get_info(position)
    ##        assert at_point is not None
    ##        if at_point.color <= 0 and row in self.additions:
    ##            tile = self.additions[row]
    ##            if isinstance(tile, int) and self.can_place_tile_color_at_point(
    ##                position,
    ##                tile,
    ##            ):
    ##                self.place_tile(position, tile)
    ##                self.additions[row] = column
    ##                # Update invalid placements after new placement
    ##                self.remove_invalid_additions()
    ##                success = True
    ##        if not self.get_rows_to_tile_map():
    ##            self.wall_tiling = False
    ##        return success

    ##    def wall_tiling_mode(self, moved_table: dict[int, int]) -> None:
    ##        """Set self into Wall Tiling Mode. Finishes automatically if not in variant play mode."""
    ##        self.wall_tiling = True
    ##        for key, value in moved_table.items():
    ##            key = int(key) - 1
    ##            if key in self.additions:
    ##                raise RuntimeError(
    ##                    f"Key {key!r} Already in additions dictionary!",
    ##                )
    ##            self.additions[key] = value
    ##        if not self.variant_play:
    ##            for row in range(self.size[1]):
    ##                if row in self.additions:
    ##                    rowdata = [tile.color for tile in self.get_row(row)]
    ##                    tile = self.additions[row]
    ##                    if not isinstance(tile, int):
    ##                        continue
    ##                    negative_tile_color = -(tile.color + 1)
    ##                    if negative_tile_color in rowdata:
    ##                        column = rowdata.index(negative_tile_color)
    ##                        self.place_tile((column, row), tile)
    ##                        # Set data to the column placed in, use for scoring
    ##                        self.additions[row] = column
    ##                    else:
    ##                        raise RuntimeError(
    ##                            f"{negative_tile_color} not in row {row}!",
    ##                        )
    ##                else:
    ##                    raise RuntimeError(f"{row} not in moved_table!")
    ##            self.wall_tiling = False
    ##        else:
    ##            # Invalid additions can only happen in variant play mode.
    ##            self.remove_invalid_additions()

    def get_touches_continuous(
        self,
        xy: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        """Return two lists, each of which contain all the tiles that touch the tile at given x y position, including that position."""
        rs, cs = self.size
        x, y = xy
        # Get row and column tile color data
        row = list(self.get_row(y))
        column = list(self.get_column(x))

        # Both
        def get_greater_than(v: int, size: int, data: list[int]) -> list[int]:
            """Go through data forward and backward from point v out by size, and return all points from data with a value >= 0."""

            def try_range(range_: Iterable[int]) -> list[int]:
                """Try range. Return all of data in range up to when indexed value is < 0."""
                ret = []
                for tv in range_:
                    if data[tv] < 0:
                        break
                    ret.append(tv)
                return ret

            nt = try_range(reversed(range(v)))
            pt = try_range(range(v + 1, size))
            return nt + pt

        def comb(one: Iterable[T], two: Iterable[RT]) -> list[tuple[T, RT]]:
            """Combine two lists by zipping together and returning list object."""
            return list(zip(one, two, strict=False))

        def get_all(lst: list[tuple[int, int]]) -> Generator[int, None, None]:
            """Return all of the self.get_info points for each value in lst."""
            for pos in lst:
                yield self.get_info(pos)

        # Get row touches
        row_touches = comb(get_greater_than(x, rs, row), [y] * rs)
        # Get column touches
        column_touches = comb([x] * cs, get_greater_than(y, cs, column))
        # Get real tiles from indexes and return
        return list(get_all(row_touches)), list(get_all(column_touches))

    def score_additions(self) -> int:
        """Return the number of points the additions scored.

        Uses self.additions, which is set in self.wall_tiling_mode()
        """
        score = 0
        for x, y in ((self.additions[y], y) for y in range(self.size[1])):
            if x is not None:
                assert isinstance(x, int)
                rowt, colt = self.get_touches_continuous((x, y))
                horiz = len(rowt)
                verti = len(colt)
                if horiz > 1:
                    score += horiz
                if verti > 1:
                    score += verti
                if horiz <= 1 and verti <= 1:
                    score += 1
            del self.additions[y]
        return score

    def get_filled_rows(self) -> int:
        """Return the number of filled rows on this board."""
        count = 0
        for row in range(self.size[1]):
            real = (t >= 0 for t in self.get_row(row))
            if all(real):
                count += 1
        return count

    def has_filled_row(self) -> bool:
        """Return True if there is at least one completely filled horizontal line."""
        return self.get_filled_rows() >= 1

    def get_filled_columns(self) -> int:
        """Return the number of filled rows on this board."""
        count = 0
        for column in range(self.size[0]):
            real = (t >= 0 for t in self.get_column(column))
            if all(real):
                count += 1
        return count

    def get_filled_colors(self) -> int:
        """Return the number of completed colors on this board."""
        color_count = Counter(
            self.get_info((x, y))
            for x in range(self.size[0])
            for y in range(self.size[1])
        )
        count = 0
        for fill_count in color_count.values():
            if fill_count >= 5:
                count += 1
        return count

    def end_of_game_scoreing(self) -> int:
        """Return the additional points for this board at the end of the game."""
        score = 0
        score += self.get_filled_rows() * 2
        score += self.get_filled_columns() * 7
        score += self.get_filled_colors() * 10
        return score


class Row(TileRenderer):
    """Represents one of the five rows each player has."""

    __slots__ = ("color", "player", "size", "tiles")
    greyshift = GREYSHIFT

    def __init__(
        self,
        size: int,
        tile_separation: int | None = None,
        background: tuple[int, int, int] | None = None,
    ) -> None:
        """Initialize row."""
        super().__init__(
            "Row",
            tile_separation,
            background,
        )
        self.size = int(size)

        self.color = Tile.blank
        self.tiles = list([self.color] * self.size)

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.size})"

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image((self.size, 1))

        for x in range(len(self.tiles)):
            self.blit_tile(self.tiles[x], (x, 0))

    def get_tile_point(self, screen_location: tuple[int, int]) -> int | None:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        pos = super().get_tile_point()
        if pos is None:
            return None
        return pos[0]

    def get_placed(self) -> int:
        """Return the number of tiles in self that are not fake tiles, like grey ones."""
        return len([tile for tile in self.tiles if tile.color >= 0])

    def get_placeable(self) -> int:
        """Return the number of tiles permitted to be placed on self."""
        return self.size - self.get_placed()

    def is_full(self) -> bool:
        """Return True if this row is full."""
        return self.get_placed() == self.size

    def get_info(self, location: int) -> int | None:
        """Return tile at location without deleting it. Return None on invalid location."""
        index = self.size - 1 - location
        if index < 0 or index > len(self.tiles):
            return None
        return self.tiles[index]

    def can_place(self, tile: int) -> bool:
        """Return True if permitted to place given tile object on self."""
        placeable = (tile.color == self.color) or (
            self.color < 0 and tile.color >= 0
        )
        if not placeable:
            return False
        color_correct = tile.color >= 0 and tile.color < 5
        if not color_correct:
            return False
        number_correct = self.get_placeable() > 0
        if not number_correct:
            return False

        board = self.player.get_object_by_name("Board")
        assert isinstance(board, Board)
        # Is color not present?
        return tile.color not in board.get_colors_in_row(
            self.size - 1,
        )

    def get_tile(self, replace: int = Tile.blank) -> int:
        """Return the leftmost tile while deleting it from self."""
        self.tiles.appendleft(int(replace))
        self.image_update = True
        return self.tiles.pop()

    def place_tile(self, tile: int) -> None:
        """Place a given int Object on self if permitted."""
        if self.can_place(tile):
            self.color = tile.color
            self.tiles.append(tile)
            end = self.tiles.popleft()
            if not end.color < 0:
                raise RuntimeError(
                    "Attempted deletion of real tile from Row!",
                )
            self.image_update = True
        else:
            raise ValueError("Not allowed to place.")

    def can_place_tiles(self, tiles: list[int]) -> bool:
        """Return True if permitted to place all of given tiles objects on self."""
        if len(tiles) > self.get_placeable():
            return False
        for tile in tiles:
            if not self.can_place(tile):
                return False
        tile_colors = []
        for tile in tiles:
            if tile.color not in tile_colors:
                tile_colors.append(tile.color)
        return not len(tile_colors) > 1

    def place_tiles(self, tiles: list[int]) -> None:
        """Place multiple tile objects on self if permitted."""
        if self.can_place_tiles(tiles):
            for tile in tiles:
                self.place_tile(tile)
        else:
            raise ValueError("Not allowed to place tiles.")

    ##    def wall_tile(
    ##        self,
    ##        add_to_table: dict[str, list[int] | int | None],
    ##        empty_color: int = Tile.blank,
    ##    ) -> None:
    ##        """Move tiles around and into add dictionary for the wall tiling phase of the game. Removes tiles from self."""
    ##        if "tiles_for_box" not in add_to_table:
    ##            add_to_table["tiles_for_box"] = []
    ##        if not self.is_full():
    ##            add_to_table[str(self.size)] = None
    ##            return
    ##        self.color = empty_color
    ##        add_to_table[str(self.size)] = self.get_tile()
    ##        for_box = add_to_table["tiles_for_box"]
    ##        assert isinstance(for_box, list)
    ##        for _i in range(self.size - 1):
    ##            for_box.append(self.get_tile())

    def set_background(self, color: tuple[int, int, int] | None) -> None:
        """Set the background color for this row."""
        self.background = color
        self.update_image()


class PatternLine(MultipartObject):
    """Represents multiple rows to make the pattern line."""

    __slots__ = ("player", "row_separation")
    size = (5, 5)

    def __init__(self, player: Player, row_separation: int = 0) -> None:
        """Initialize pattern line."""
        super().__init__("PatternLine")
        self.player = player
        self.row_separation = row_separation

        for x, _y in zip(
            range(self.size[0]),
            range(self.size[1]),
            strict=True,
        ):
            self.add_object(Row(self.player, x + 1))

        self.set_background(None)

        self._lastloc = Vector2(0, 0)

    def set_background(self, color: tuple[int, int, int] | None) -> None:
        """Set the background color for all rows in the pattern line."""
        self.set_attr_all("back", color)
        self.set_attr_all("image_update", True)

    def get_row(self, row: int) -> Row:
        """Return given row."""
        object_ = self.get_object(row)
        assert isinstance(object_, Row)
        return object_

    def reset_position(self) -> None:
        """Reset Locations of Rows according to self.location."""
        last = self.size[1]
        w = self.get_row(last - 1).width_height[0]
        if w is None:
            raise RuntimeError(
                "Image Dimensions for Row Object (row.width_height) are None!",
            )
        h1 = self.get_row(0).tile_full
        h = int(last * h1)
        self.width_height = w, h
        w1 = h1 / 2

        x, y = self.location
        y -= h / 2 - w1
        for rid in self.objects:
            row = self.get_row(rid)
            diff = last - row.size
            row.location = Vector2(x + (diff * w1), y + rid * h1)

    def get_tile_point(
        self,
        screen_location: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        for y in range(self.size[1]):
            x = self.get_row(y).get_tile_point(screen_location)
            if x is not None:
                return x, y
        return None

    def is_full(self) -> bool:
        """Return True if self is full."""
        return all(self.get_row(rid).is_full() for rid in range(self.size[1]))

    def wall_tiling(self) -> dict[str, list[int] | int | None]:
        """Return a dictionary to be used with wall tiling. Removes tiles from rows."""
        values: dict[str, list[int] | int | None] = {}
        for rid in range(self.size[1]):
            self.get_row(rid).wall_tile(values)
        return values

    def process(self, time_passed: float) -> None:
        """Process all the rows that make up the pattern line."""
        if self.hidden != self._lasthidden:
            self.set_attr_all("image_update", True)
        super().process(time_passed)


class FloorLine(Row):
    """Represents a player's floor line."""

    size = 7
    number_one_color = Tile.one

    def __init__(self, player: Player) -> None:
        """Initialize floor line."""
        super().__init__(player, self.size, background=ORANGE)
        self.name = "floor_line"

        # self.font = Font(FONT, round(self.tile_size*1.2), color=BLACK, cx=False, cy=False)
        self.text = Text(
            round(self.tile_size * 1.2),
            BLACK,
            cx=False,
            cy=False,
        )
        self.has_number_one_tile = False

        gen = floor_line_subtract_generator(1)
        self.numbers = [next(gen) for i in range(self.size)]

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

    # self.font.render(surface, str(self.numbers[x]), xy)

    def place_tile(self, tile: int) -> None:
        """Place a given int Object on self if permitted."""
        self.tiles.insert(self.get_placed(), tile)

        if tile.color == self.number_one_color:
            self.has_number_one_tile = True

        box_lid = self.player.game.get_object_by_name("BoxLid")
        assert isinstance(box_lid, BoxLid)

        def handle_end(end: int) -> None:
            """Handle the end tile we are replacing. Ensures number one tile is not removed."""
            if not end.color < 0:
                if end.color == self.number_one_color:
                    handle_end(self.tiles.pop())
                    self.tiles.appendleft(end)
                    return
                box_lid.add_tile(end)

        handle_end(self.tiles.pop())

        self.image_update = True

    def score_tiles(self) -> int:
        """Score self.tiles and return how to change points."""
        running_total = 0
        for x in range(self.size):
            if self.tiles[x].color >= 0:
                running_total += self.numbers[x]
            elif x < self.size - 1 and self.tiles[x + 1].color >= 0:
                raise RuntimeError(
                    "Player is likely cheating! Invalid placement of floor_line tiles!",
                )
        return running_total

    def get_tiles(
        self,
        empty_color: int = Tile.blank,
    ) -> tuple[list[int], int | None]:
        """Return tuple of tiles gathered, and then either the number one tile or None."""
        tiles = []
        number_one_tile = None
        for tile in (self.tiles.pop() for i in range(len(self.tiles))):
            if tile.color == self.number_one_color:
                number_one_tile = tile
                self.has_number_one_tile = False
            elif tile.color >= 0:
                tiles.append(tile)

        for _i in range(self.size):
            self.tiles.append(int(empty_color))
        self.image_update = True
        return tiles, number_one_tile

    def can_place_tiles(self, tiles: list[int]) -> bool:
        """Return True."""
        return True


class Factory(Grid):
    """Represents a Factory."""

    size = (2, 2)
    color = WHITE
    outline = BLUE
    out_size = 0.1

    def __init__(self, factory_id: int) -> None:
        """Initialize factory."""
        super().__init__(self.size, background=None)
        self.number = factory_id
        self.name = f"Factory{self.number}"

        self.radius = math.ceil(
            self.tile_full * self.size[0] * self.size[1] / 3 + 3,
        )

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.number})"

    def add_circle(self, surface: pygame.surface.Surface) -> None:
        """Add circle to self.image."""
        rad = math.ceil(self.radius)
        surf = pygame.surface.Surface((2 * rad, 2 * rad), SRCALPHA)
        pygame.draw.circle(surf, self.outline, (rad, rad), rad)
        pygame.draw.circle(
            surf,
            self.color,
            (rad, rad),
            math.ceil(rad * (1 - self.out_size)),
        )

        surface.blit(
            surf,
            (
                round(self.location[0] - self.radius),
                round(self.location[1] - self.radius),
            ),
        )

    def render(self, surface: pygame.surface.Surface) -> None:
        """Render Factory."""
        if not self.hidden:
            self.add_circle(surface)
        super().render(surface)

    def fill(self, tiles: list[int]) -> None:
        """Fill self with tiles. Will raise exception if insufficiant tiles."""
        if len(tiles) < self.size[0] * self.size[1]:
            size = self.size[0] * self.size[1]
            raise RuntimeError(
                f"Insufficiant quantity of tiles! Needs {size}!",
            )
        for y in range(self.size[1]):
            for tile, x in zip(
                (tiles.pop() for i in range(self.size[0])),
                range(self.size[0]),
                strict=True,
            ):
                self.place_tile((x, y), tile)
        if tiles:
            raise RuntimeError("Too many tiles!")

    def grab(self) -> list[int]:
        """Return all tiles on this factory."""
        return [
            tile
            for tile in (
                self.get_tile((x, y), Tile.blank)
                for x in range(self.size[0])
                for y in range(self.size[1])
            )
            if tile is not None and tile.color != Tile.blank
        ]

    def grab_color(self, color: int) -> tuple[list[int], list[int]]:
        """Return all tiles of color given in the first list, and all non-matches in the second list."""
        tiles = self.grab()
        right, wrong = [], []
        for tile in tiles:
            if tile.color == color:
                right.append(tile)
            else:
                wrong.append(tile)
        return right, wrong

    def process(self, time_passed: float) -> None:
        """Process self."""
        if self.image_update:
            self.radius = int(
                self.tile_full * self.size[0] * self.size[1] // 3 + 3,
            )
        super().process(time_passed)


class Factories(MultipartObject):
    """Factories Multipart Object, made of multiple Factory Objects."""

    tiles_each = 4

    def __init__(
        self,
        game: Game,
        factories: int,
        size: int | None = None,
    ) -> None:
        """Initialize factories."""
        super().__init__("Factories")

        self.game = game
        self.count = factories

        for i in range(self.count):
            self.add_object(Factory(self.game, i))

        if size is None:
            factory = self.objects[0]
            assert isinstance(factory, Factory)
            factory.process(0)
            rad = factory.radius
            self.size = rad * 5
        else:
            self.size = size
        self.size = math.ceil(self.size)

        self.play_tiles_from_bag()

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}(%r, %i, ...)" % (
            self.game,
            self.count,
        )

    def reset_position(self) -> None:
        """Reset the position of all factories within."""
        for index, degrees in enumerate(range(0, 360, 360 // self.count)):
            self.objects[index].location = (
                Vector2.from_degrees(
                    degrees,
                    self.size,
                )
                + self.location
            )

    def process(self, time_passed: float) -> None:
        """Process factories. Does not react to cursor if hidden."""
        super().process(time_passed)
        if self.hidden:
            return
        cursor = self.game.get_object_by_name("Cursor")
        assert isinstance(cursor, Cursor)
        if not cursor.is_pressed() or cursor.is_holding():
            return
        obj, point = self.get_intersection(cursor.location)
        if obj is None or point is None:
            return
        oid = int(obj[7:])

        factory = self.objects[oid]
        assert isinstance(factory, Factory)

        tile_at_point = factory.get_info(point)
        if tile_at_point is None or tile_at_point.color < 0:
            return
        table = self.game.get_object_by_name("TableCenter")
        assert isinstance(table, TableCenter)
        select, tocenter = factory.grab_color(
            tile_at_point.color,
        )
        if tocenter:
            table.add_tiles(tocenter)
        cursor.drag(select)

    def play_tiles_from_bag(self, empty_color: int = Tile.blank) -> None:
        """Divy up tiles to each factory from the bag."""
        # For every factory we have,
        for fid in range(self.count):
            # Draw tiles for the factory
            drawn = []
            for _i in range(self.tiles_each):
                # If the bag is not empty,
                if not self.game.bag.is_empty():
                    # Draw a tile from the bag.
                    tile = self.game.bag.draw_tile()
                    assert tile is not None
                    drawn.append(tile)
                else:  # Otherwise, get the box lid
                    box_lid = self.game.get_object_by_name("BoxLid")
                    assert isinstance(box_lid, BoxLid)
                    # If the box lid is not empty,
                    if not box_lid.is_empty():
                        # Add all the tiles from the box lid to the bag
                        self.game.bag.add_tiles(box_lid.get_tiles())
                        # and shake the bag to randomize everything
                        self.game.bag.reset()
                        # Then, grab a tile from the bag like usual.
                        tile = self.game.bag.draw_tile()
                        assert tile is not None
                        drawn.append(tile)
                    else:
                        # "In the rare case that you run out of tiles again
                        # while there are none left in the lid, start a new
                        # round as usual even though are not all factory
                        # displays are properly filled."
                        drawn.append(int(empty_color))
            # Place drawn tiles on factory
            factory = self.objects[fid]
            assert isinstance(factory, Factory)
            factory.fill(drawn)

    def is_all_empty(self) -> bool:
        """Return True if all factories are empty."""
        for fid in range(self.count):
            factory = self.objects[fid]
            assert isinstance(factory, Factory)
            if not factory.is_empty():
                return False
        return True


class TableCenter(TileRenderer):
    """Object that represents the center of the table."""

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

    def update_image(self):
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


##    def process(self, time_passed: float) -> None:
##        """Process factories."""
##        if self.hidden:
##            super().process(time_passed)
##            return
##        cursor = self.game.get_object_by_name("Cursor")
##        assert isinstance(cursor, Cursor)
##        if (
##            cursor.is_pressed()
##            and not cursor.is_holding()
##            and not self.is_empty()
##            and self.is_selected(cursor.location)
##        ):
##            point = self.get_tile_point(cursor.location)
##            # Shouldn't return none anymore since we have is_selected now.
##            assert point is not None
##            tile = self.get_info(point)
##            assert isinstance(tile, int)
##            color_at_point = tile.color
##            if color_at_point >= 0 and color_at_point < 5:
##                cursor.drag(self.pull_tiles(color_at_point))
##        super().process(time_passed)


class Player(sprite.Sprite):
    """Represents a player. Made of lots of objects."""

    def __init__(
        self,
        game: Game,
        player_id: int,
        networked: bool = False,
        varient_play: bool = False,
    ) -> None:
        """Initialize player."""
        super().__init__(f"Player{player_id}")

        self.game = game
        self.player_id = player_id
        self.networked = networked
        self.varient_play = varient_play

        self.add_object(Board(self.varient_play))
        self.add_object(PatternLine(self))
        self.add_object(FloorLine(self))
        ##        self.add_object(objects.Text(SCOREFONTSIZE, SCORECOLOR))

        self.score = 0
        self.is_turn = False
        self.is_wall_tiling = False
        self.just_held = False
        self.just_dropped = False

        self.update_score()

        self._lastloc = Vector2(0, 0)

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}(%r, %i, %s, %s)" % (
            self.game,
            self.player_id,
            self.networked,
            self.varient_play,
        )

    ##    def update_score(self) -> None:
    ##        """Update the scorebox for this player."""
    ##        score_box = self.get_object_by_name("Text")
    ##        assert isinstance(score_box, Text)
    ##        score_box.update_value(f"Player {self.player_id + 1}: {self.score}")

    def trigger_turn_now(self) -> None:
        """Handle start of turn."""
        if not self.is_turn:
            pattern_line = self.get_object_by_name("PatternLine")
            assert isinstance(pattern_line, PatternLine)
            if self.is_wall_tiling:
                board = self.get_object_by_name("Board")
                assert isinstance(board, Board)
                rows = board.get_rows_to_tile_map()
                for rowpos, value in rows.items():
                    color = get_tile_color(value, board.greyshift)
                    assert isinstance(color[0], int)
                    pattern_line.get_row(rowpos).set_background(
                        color,
                    )
            else:
                pattern_line.set_background(PATSELECTCOLOR)
        self.is_turn = True

    def end_of_turn(self) -> None:
        """Handle end of turn."""
        if self.is_turn:
            pattern_line = self.get_object_by_name("PatternLine")
            assert isinstance(pattern_line, PatternLine)
            pattern_line.set_background(None)
        self.is_turn = False

    ##    def end_of_game_trigger(self) -> None:
    ##        """Handle end of game.
    ##
    ##        Called by end state when game is over
    ##        Hide pattern lines and floor line.
    ##        """
    ##        pattern = self.get_object_by_name("PatternLine")
    ##        floor = self.get_object_by_name("floor_line")
    ##
    ##        pattern.hidden = True
    ##        floor.hidden = True

    def reset_position(self) -> None:
        """Reset positions of all parts of self based off self.location."""
        x, y = self.location

        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)
        bw, bh = board.width_height
        board.location = Vector2(x + bw // 2, y)

        pattern_line = self.get_object_by_name("PatternLine")
        assert isinstance(pattern_line, PatternLine)
        lw = pattern_line.width_height[0] // 2
        pattern_line.location = Vector2(x - lw, y)

        floor_line = self.get_object_by_name("floor_line")
        assert isinstance(floor_line, FloorLine)
        floor_line.location = Vector2(
            int(x - lw * (2 / 3) + TILESIZE / 3.75),
            int(y + bh * (2 / 3)),
        )

        text = self.get_object_by_name("Text")
        assert isinstance(text, Text)
        text.location = Vector2(x - (bw // 3), y - (bh * 2 // 3))


##    def wall_tiling(self) -> None:
##        """Do the wall tiling phase of the game for this player."""
##        self.is_wall_tiling = True
##        pattern_line = self.get_object_by_name("PatternLine")
##        assert isinstance(pattern_line, PatternLine)
##        board = self.get_object_by_name("Board")
##        assert isinstance(board, Board)
##        box_lid = self.game.get_object_by_name("BoxLid")
##        assert isinstance(box_lid, BoxLid)
##
##        data = pattern_line.wall_tiling()
##        tiles_for_box = data["tiles_for_box"]
##        assert isinstance(tiles_for_box, list)
##        box_lid.add_tiles(tiles_for_box)
##        del data["tiles_for_box"]
##
##        cleaned = {}
##        for key, value in data.items():
##            if not isinstance(value, int):
##                continue
##            cleaned[int(key)] = value
##
##        board.wall_tiling_mode(cleaned)

##    def done_wall_tiling(self) -> bool:
##        """Return True if internal Board is done wall tiling."""
##        board = self.get_object_by_name("Board")
##        assert isinstance(board, Board)
##        return not board.is_wall_tiling()

##    def next_round(self) -> None:
##        """Handle end of wall tiling."""
##        self.is_wall_tiling = False

##    def score_phase(self) -> int | None:
##        """Do the scoring phase of the game for this player. Return number one tile or None."""
##        board = self.get_object_by_name("Board")
##        floor_line = self.get_object_by_name("floor_line")
##        box_lid = self.game.get_object_by_name("BoxLid")
##        assert isinstance(board, Board)
##        assert isinstance(floor_line, FloorLine)
##        assert isinstance(box_lid, BoxLid)
##
##        def saturatescore() -> None:
##            if self.score < 0:
##                self.score = 0
##
##        self.score += board.score_additions()
##        self.score += floor_line.score_tiles()
##        saturatescore()
##
##        tiles_for_box, number_one = floor_line.get_tiles()
##        box_lid.add_tiles(tiles_for_box)
##
##        self.update_score()
##
##        return number_one

##    def end_of_game_scoring(self) -> None:
##        """Update final score with additional end of game points."""
##        board = self.get_object_by_name("Board")
##        assert isinstance(board, Board)
##
##        self.score += board.end_of_game_scoreing()
##
##        self.update_score()

##    def has_horzontal_line(self) -> bool:
##        """Return True if this player has a horizontal line on their game board filled."""
##        board = self.get_object_by_name("Board")
##        assert isinstance(board, Board)
##
##        return board.has_filled_row()

##    def get_horizontal_lines(self) -> int:
##        """Return the number of filled horizontal lines this player has on their game board."""
##        board = self.get_object_by_name("Board")
##        assert isinstance(board, Board)
##
##        return board.get_filled_rows()

##    def process(self, time_passed: float) -> None:
##        """Process Player."""
##        if not self.is_turn:  # Is our turn?
##            self.set_attr_all("hidden", self.hidden)
##            super().process(time_passed)
##            return
##        if self.hidden and self.is_wall_tiling and self.varient_play:
##            # If hidden, not anymore. Our turn.
##            self.hidden = False
##        if self.networked:  # We are networked.
##            self.set_attr_all("hidden", self.hidden)
##            super().process(time_passed)
##            return
##
##        cursor = self.game.get_object_by_name("Cursor")
##        assert isinstance(cursor, Cursor)
##        box_lid = self.game.get_object_by_name("BoxLid")
##        assert isinstance(box_lid, BoxLid)
##        pattern_line = self.get_object_by_name("PatternLine")
##        assert isinstance(pattern_line, PatternLine)
##        floor_line = self.get_object_by_name("floor_line")
##        assert isinstance(floor_line, FloorLine)
##        board = self.get_object_by_name("Board")
##        assert isinstance(board, Board)
##
##        if not cursor.is_pressed():
##            # Mouse up
##            if self.just_held:
##                self.just_held = False
##            if self.just_dropped:
##                self.just_dropped = False
##            self.set_attr_all("hidden", self.hidden)
##            super().process(time_passed)
##            return
##
##        # Mouse down
##        obj, point = self.get_intersection(cursor.location)
##        if obj is None or point is None:
##            if self.is_wall_tiling and self.done_wall_tiling():
##                self.next_round()
##                self.game.next_turn()
##            self.set_attr_all("hidden", self.hidden)
##            super().process(time_passed)
##            return
##        # Something pressed
##        if cursor.is_holding():  # Cursor holding tiles
##            move_made = False
##            if not self.is_wall_tiling:  # Is wall tiling:
##                if obj == "PatternLine":
##                    pos, row_number = point
##                    row = pattern_line.get_row(row_number)
##                    if not row.is_full():
##                        info = row.get_info(pos)
##                        if info is not None and info.color < 0:
##                            _color, _held = cursor.get_held_info()
##                            todrop = min(
##                                pos + 1,
##                                row.get_placeable(),
##                            )
##                            tiles = cursor.drop(todrop)
##                            if row.can_place_tiles(tiles):
##                                row.place_tiles(tiles)
##                                move_made = True
##                            else:
##                                cursor.force_hold(tiles)
##                elif obj == "floor_line":
##                    tiles_to_add = cursor.drop()
##                    if floor_line.is_full():
##                        # Floor is full,
##                        # Add tiles to box instead.
##                        box_lid.add_tiles(tiles_to_add)
##                    elif floor_line.get_placeable() < len(
##                        tiles_to_add,
##                    ):
##                        # Floor is not full but cannot fit all in floor line.
##                        # Add tiles to floor line and then to box
##                        while len(tiles_to_add) > 0:
##                            if floor_line.get_placeable() > 0:
##                                floor_line.place_tile(
##                                    tiles_to_add.pop(),
##                                )
##                            else:
##                                box_lid.add_tile(
##                                    tiles_to_add.pop(),
##                                )
##                    else:
##                        # Otherwise add to floor line for all.
##                        floor_line.place_tiles(tiles_to_add)
##                    move_made = True
##            elif not self.just_held and obj == "Board":
##                tile = board.get_info(point)
##                assert isinstance(tile, int)
##                if tile.color == Tile.blank:
##                    # Cursor holding and wall tiling
##                    _column, row_id = point
##                    cursor_tile = cursor.drop(1)[0]
##                    board_tile = board.get_tile_for_cursor_by_row(
##                        row_id,
##                    )
##                    if (
##                        board_tile is not None
##                        and cursor_tile.color == board_tile.color
##                        and board.wall_tile_from_point(point)
##                    ):
##                        self.just_dropped = True
##                        pattern_line.get_row(
##                            row_id,
##                        ).set_background(None)
##            if move_made and not self.is_wall_tiling:
##                if cursor.holding_number_one:
##                    one_tile = cursor.drop_one_tile()
##                    assert one_tile is not None
##                    floor_line.place_tile(one_tile)
##                if cursor.get_held_count(True) == 0:
##                    self.game.next_turn()
##        elif self.is_wall_tiling and obj == "Board" and not self.just_dropped:
##            # Mouse down, something pressed, and not holding anything
##            # Wall tiling, pressed, not holding
##            _column_number, row_number = point
##            tile = board.get_tile_for_cursor_by_row(
##                row_number,
##            )
##            if tile is not None:
##                cursor.drag([tile])
##                self.just_held = True
##        if self.is_wall_tiling and self.done_wall_tiling():
##            self.next_round()
##            self.game.next_turn()
##        self.set_attr_all("hidden", self.hidden)
##        super().process(time_passed)


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
    """Outlined Text with attributes settable via keyword arguments."""

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
    """Button with attributes settable via keyword arguments."""

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
        """Add a new Button object to group."""
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
        """Add a new Text object to self.game with arguments. Return text id."""
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
            text = self.game.get_object_by_name(f"Text{text_name}")
            assert isinstance(text, Text)
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


class InitializeState(GameState):
    """Initialize state."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize self."""
        super().__init__("initialize")

    async def entry_actions(self) -> None:
        """Set up buttons."""
        assert self.machine is not None
        self.id = self.machine.new_group("initialize")

        self.group_add(Cursor())
        await self.manager.raise_event(Event("cursor_drag", [3, 5]))
        self.manager.register_handler("PygameMouseMotion", self.mouse_moved)

        ##        board = Board()
        ####        board.place_tile((2, 2), Tile.red)
        ##        board.location = Vector2.from_iter(SCREEN_SIZE) // 2
        ##        self.group_add(board)

        center = TableCenter()
        center.location = Vector2.from_iter(SCREEN_SIZE) // 2
        self.group_add(center)
        center.add_tiles((0, 1, 2, 3, 5))

    async def mouse_moved(
        self,
        event: Event[sprite.PygameMouseMotion],
    ) -> None:
        ##        print(f'{event = }')
        await self.manager.raise_event(
            Event("cursor_set_location", event.data["pos"]),
        )


##        await self.manager.raise_event(
##            Event("cursor_set_destination", event.data["pos"]),
##        )


##    async def check_conditions(self) -> str:
##        """Go to title state."""
##        return "title"


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
            handle_click=self.change_state("play_internal_hosting"),
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

    def entry_actions(self) -> None:
        """Add cursor object and tons of button and text objects to the game."""
        super().entry_actions()

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

    def entry_actions(self) -> None:
        """Advance turn."""
        assert self.game is not None
        self.game.next_turn()

    def check_state(self) -> str | None:
        """If all tiles are gone, go to wall tiling. Otherwise keep waiting for that to happen."""
        assert self.game is not None
        fact = self.game.get_object_by_name("Factories")
        assert isinstance(fact, Factories)
        table = self.game.get_object_by_name("TableCenter")
        assert isinstance(table, TableCenter)
        cursor = self.game.get_object_by_name("Cursor")
        assert isinstance(cursor, Cursor)
        if (
            fact.is_all_empty()
            and table.is_empty()
            and not cursor.is_holding(True)
        ):
            return "WallTiling"
        return None


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

    def do_actions(self) -> None:
        """Perform actions of state."""
        assert self.game is not None
        if self.new_round:
            fact = self.game.get_object_by_name("Factories")
            assert isinstance(fact, Factories)
            # This also handles bag re-filling from box lid.
            fact.play_tiles_from_bag()

    def check_state(self) -> str:
        """Go to factory offer if new round else end screen."""
        if self.new_round:
            return "FactoryOffer"
        return "End"


class EndScreen(MenuState):
    """End screen state."""

    def __init__(self) -> None:
        """Initialize end screen."""
        super().__init__("End")
        self.ranking: dict[int, list[int]] = {}
        self.wininf = ""

    def get_winners(self) -> None:
        """Update self.ranking by player scores."""
        assert self.game is not None
        self.ranking.clear()
        scpid = {}
        for player_id in range(self.game.players):
            player = self.game.get_player(player_id)
            assert isinstance(player, Player)
            player.end_of_game_trigger()
            if player.score not in scpid:
                scpid[player.score] = [player_id]
            else:
                scpid[player.score] += [player_id]
        # make sure no ties and establish rank
        rank = 1
        for score in sorted(scpid, reverse=True):
            pids = scpid[score]
            if len(pids) > 1:
                # If players have same score,
                # most horizontal lines is tie breaker.
                players = [
                    self.game.get_player(player_id) for player_id in pids
                ]
                lines = [
                    (p.get_horizontal_lines(), p.player_id) for p in players
                ]
                last = None
                for c, player_id in sorted(
                    lines,
                    key=operator.itemgetter(0),
                    reverse=True,
                ):
                    if last == c:
                        self.ranking[rank - 1] += [player_id + 1]
                        continue
                    last = c
                    self.ranking[rank] = [player_id + 1]
                    rank += 1
            else:
                self.ranking[rank] = [pids[0] + 1]
                rank += 1
        # Finally, make nice text.
        text = ""
        for rank in sorted(self.ranking):
            line = "Player"
            players_rank = self.ranking[rank]
            cnt = len(players_rank)
            if cnt > 1:
                line += "s"
            line += " "
            if cnt == 1:
                line += "{}"
            elif cnt == 2:
                line += "{} and {}"
            elif cnt >= 3:
                tmp = (["{}"] * (cnt - 1)) + ["and {}"]
                line += ", ".join(tmp)
            line += " "
            if cnt == 1:
                line += "got"
            else:
                line += "tied for"
            line += " "
            if rank <= 2:
                line += ("1st", "2nd")[rank - 1]
            else:
                line += f"{rank}th"
            line += " place!\n"
            text += line.format(*players_rank)
        self.wininf = text[:-1]

    def entry_actions(self) -> None:
        """Set up end screen."""
        assert self.game is not None
        # Figure out who won the game by points.
        self.get_winners()
        # Hide everything
        table = self.game.get_object_by_name("TableCenter")
        assert isinstance(table, TableCenter)
        table.hidden = True

        fact = self.game.get_object_by_name("Factories")
        assert isinstance(fact, Factories)
        fact.set_attr_all("hidden", True)

        # Add buttons
        bid = self.add_button(
            "ReturnTitle",
            "Return to Title",
            self.to_state("Title"),
            (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] * 4 // 5),
        )
        buttontitle = self.game.get_object(bid)
        assert isinstance(buttontitle, Button)
        buttontitle.Render_Priority = "last-1"
        buttontitle.cur_time = 2

        # Add score board
        x = SCREEN_SIZE[0] // 2
        y = 10
        for idx, line in enumerate(self.wininf.split("\n")):
            self.add_text(f"Line{idx}", line, (x, y), cx=True, cy=False)
            # self.game.get_object(bid).Render_Priority = f'last{-(2+idx)}'
            button = self.game.get_object(bid)
            assert isinstance(button, Button)
            button.Render_Priority = "last-2"
            y += self.bh


class Game(ObjectHandler):
    """Game object, contains most of what's required for Azul."""

    tile_size = 30

    def __init__(self) -> None:
        """Initialize game."""
        super().__init__()
        # Gets overwritten by Keyboard object
        self.keyboard: Keyboard | None = None

        self.states: dict[str, GameState] = {}
        self.active_state: GameState | None = None

        self.add_states(
            [
                InitializeState(),
                TitleState(),
                CreditsState(),
                SettingsState(),
                PhaseFactoryOffer(),
                PhaseWallTiling(),
                PhasePrepareNext(),
                EndScreen(),
                ##                PhaseFactoryOfferNetworked(),
                ##                PhaseWallTilingNetworked(),
                ##                PhasePrepareNextNetworked(),
                ##                EndScreenNetworked(),
            ],
        )
        self.initialized_state = False

        self.background_color = BACKGROUND

        self.is_host = True
        self.players = 0
        self.factories = 0

        self.player_turn: int = 0

        # Tiles
        self.bag = Bag(TILECOUNT, REGTILECOUNT)

    # # Cache
    # self.cache: dict[int, pygame.surface.Surface] = {}

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}()"

    def add_object(self, obj: Object) -> None:
        """Add an object to the game."""
        obj.game = self
        super().add_object(obj)

    def get_player(self, player_id: int) -> Player:
        """Get the player with player id player_id."""
        if self.players:
            player = self.get_object_by_name(f"Player{player_id}")
            assert isinstance(player, Player)
            return player
        raise RuntimeError("No players!")

    def player_turn_over(self) -> None:
        """Call end_of_turn for current player."""
        if self.player_turn >= 0 and self.player_turn < self.players:
            old_player = self.get_player(self.player_turn)
            if old_player.is_turn:
                old_player.end_of_turn()

    def next_turn(self) -> None:
        """Tell current player it's the end of their turn, and update who's turn it is and now it's their turn."""
        if self.is_host:
            self.player_turn_over()
            last = self.player_turn
            self.player_turn = (self.player_turn + 1) % self.players
            if self.player_turn == last and self.players > 1:
                self.next_turn()
                return
            new_player = self.get_player(self.player_turn)
            new_player.trigger_turn_now()

    def start_game(
        self,
        players: int,
        varient_play: bool = False,
        host_mode: bool = True,
        address: str = "",
    ) -> None:
        """Start a new game."""
        self.reset_cache()
        max_players = 4
        self.players = saturate(players, 1, max_players)
        self.is_host = host_mode
        self.factories = self.players * 2 + 1

        self.rm_star()

        self.add_object(Cursor(self))
        self.add_object(TableCenter(self))
        self.add_object(BoxLid(self))

        if self.is_host:
            self.bag.reset()
            # S311 Standard pseudo-random generators are not suitable for cryptographic purposes
            self.player_turn = random.randint(  # noqa: S311
                -1,
                self.players - 1,
            )
        else:
            raise NotImplementedError()

        cx, cy = SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2
        out = math.sqrt(cx**2 + cy**2) // 3 * 2

        mdeg = 360 // max_players

        for player_id in range(self.players):
            networked = False
            newp = Player(self, player_id, networked, varient_play)

            truedeg = (self.players + 1 - player_id) * (360 / self.players)
            closedeg = truedeg // mdeg * mdeg + 45
            rad = math.radians(closedeg)

            newp.location = Vector2(
                round(cx + out * math.sin(rad)),
                round(
                    cy + out * math.cos(rad),
                ),
            )
            self.add_object(newp)
        if self.is_host:
            self.next_turn()

        factory = Factories(self, self.factories)
        factory.location = Vector2(cx, cy)
        self.add_object(factory)
        self.process_objects(0)

        if self.is_host:
            self.next_turn()

    def screen_size_update(self) -> None:
        """Handle screen size updates."""
        objs_with_attr = self.get_objects_with_attr("screen_size_update")
        for oid in objs_with_attr:
            obj = self.get_object(oid)
            assert obj is not None
            obj.screen_size_update()


class PlayHostingState(AsyncState["AzulClient"]):
    """Start running server."""

    __slots__ = ("address",)

    internal_server = False

    def __init__(self) -> None:
        """Initialize Play internal hosting / hosting State."""
        extra = "_internal" if self.internal_server else ""
        super().__init__(f"play{extra}_hosting")

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


class PlayState(GameState):
    """Game Play State."""

    __slots__ = ("exit_data",)

    def __init__(self) -> None:
        """Initialize Play State."""
        super().__init__("play")

        # (0: normal | 1: error) <error message> <? handled>
        self.exit_data: tuple[int, str, bool] | None = None

    def register_handlers(self) -> None:
        """Register event handlers."""
        self.manager.register_handlers(
            {
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
        self.exit_data = None

        assert self.machine is not None
        if self.id == 0:
            self.id = self.machine.new_group("play")

        # self.group_add(())
        ##        gameboard = GameBoard(
        ##            45,
        ##        )
        ##        gameboard.location = [x // 2 for x in SCREEN_SIZE]
        ##        self.group_add(gameboard)

        await self.machine.raise_event(Event("init", None))

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
