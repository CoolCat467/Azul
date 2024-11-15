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

import importlib
import math
import operator
import os
import random
import time
from collections import Counter, deque
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, TypeVar, cast

import pygame
from numpy import array, int8
from pygame.locals import (
    KEYDOWN,
    KEYUP,
    QUIT,
    RESIZABLE,
    SRCALPHA,
    USEREVENT,
    VIDEORESIZE,
)
from pygame.rect import Rect

from azul.tools import (
    floor_line_subtract_generator,
    gen_random_proper_seq,
    lerp_color,
    randomize,
    saturate,
    sort_tiles,
)
from azul.vector import Vector2

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence

    from typing_extensions import TypeVarTuple, Unpack

    P = TypeVarTuple("P")

T = TypeVar("T")
RT = TypeVar("RT")

SCREENSIZE = (650, 600)

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
NUMBERONETILE = 5
TILESIZE = 15

# Colors
BACKGROUND = (0, 192, 16)
TILEDEFAULT = ORANGE
SCORECOLOR = BLACK
PATSELECTCOLOR = DARKGREEN
BUTTONTEXTCOLOR = DARKCYAN
BUTTONBACKCOLOR = WHITE
GREYSHIFT = 0.75  # 0.65

# Font
FONT: Final = FONT_FOLDER / "RuneScape-UF-Regular.ttf"
SCOREFONTSIZE = 30
BUTTONFONTSIZE = 60


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
        if tile_color == -6:
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
        if tile_color == -6:
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


@lru_cache
def get_tile_image(
    tile: Tile,
    tilesize: int,
    greyshift: float = GREYSHIFT,
    outline_size: float = 0.2,
) -> pygame.surface.Surface:
    """Return a surface of a given tile."""
    cid = tile.color
    if cid < 5:
        color = get_tile_color(cid, greyshift)

    elif cid >= 5:
        color_data = tile_colors[cid]
        assert len(color_data) == 2
        color, outline = color_data
        surf = outline_rectangle(
            make_square_surf(color, tilesize),
            outline,
            outline_size,
        )
        # Add tile symbol
        add_symbol_to_tile_surf(surf, cid, tilesize, greyshift)

        return surf
    surf = make_square_surf(color, tilesize)
    # Add tile symbol
    add_symbol_to_tile_surf(surf, cid, tilesize, greyshift)
    return surf


def set_alpha(
    surface: pygame.surface.Surface,
    alpha: int,
) -> pygame.surface.Surface:
    """Return a surface by replacing the alpha channel of it with given alpha value, preserve color."""
    surface = surface.copy().convert_alpha()
    w, h = surface.get_size()
    for y in range(h):
        for x in range(w):
            r, g, b = cast("tuple[int, int, int]", surface.get_at((x, y))[:3])
            surface.set_at((x, y), pygame.Color(r, g, b, alpha))
    return surface


def get_tile_container_image(
    wh: tuple[int, int],
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
    image = pygame.surface.Surface(wh, flags=SRCALPHA)
    if back is not None:
        image.fill(back)
    else:
        image.fill((0, 0, 0, 0))
    return image


class Font:
    """Font object, simplify using text."""

    def __init__(
        self,
        font_name: str | Path,
        fontsize: int = 20,
        color: tuple[int, int, int] = (0, 0, 0),
        cx: bool = True,
        cy: bool = True,
        antialias: bool = False,
        background: tuple[int, int, int] | None = None,
        do_cache: bool = True,
    ) -> None:
        """Initialize font."""
        self.font = font_name
        self.size = int(fontsize)
        self.color = color
        self.center = [cx, cy]
        self.antialias = bool(antialias)
        self.background = background
        self.do_cache = bool(do_cache)
        self.cache: pygame.surface.Surface | None = None
        self.last_text: str | None = None
        self._change_font()

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}(%r, %i, %r, %r, %r, %r, %r, %r)" % (
            self.font,
            self.size,
            self.color,
            self.center[0],
            self.center[1],
            self.antialias,
            self.background,
            self.do_cache,
        )

    def _change_font(self) -> None:
        """Set self.pyfont to a new pygame.font.Font object from data we have."""
        self.pyfont = pygame.font.Font(self.font, self.size)

    def _cache(self, surface: pygame.surface.Surface) -> None:
        """Set self.cache to surface."""
        self.cache = surface

    def get_height(self) -> int:
        """Return the height of font."""
        return self.pyfont.get_height()

    def render_nosurf(
        self,
        text: str | None,
        size: int | None = None,
        color: tuple[int, int, int] | None = None,
        background: tuple[int, int, int] | None = None,
        force_update: bool = False,
    ) -> pygame.surface.Surface:
        """Render and return a surface of given text. Use stored data to render, if arguments change internal data and render."""
        update_cache = (
            self.cache is None or force_update or text != self.last_text
        )
        # Update internal data if new values given
        if size is not None:
            self.size = int(size)
            self._change_font()
            update_cache = True
        if color is not None:
            self.color = color
            update_cache = True
        if self.background != background:
            self.background = background
            update_cache = True

        if self.do_cache:
            if update_cache:
                self.last_text = text
                surf = self.pyfont.render(
                    text,
                    self.antialias,
                    self.color,
                    self.background,
                ).convert_alpha()
                self._cache(surf.copy())
            else:
                assert self.cache is not None
                surf = self.cache
        else:
            # Render the text using the pygame font
            surf = self.pyfont.render(
                text,
                self.antialias,
                self.color,
                self.background,
            ).convert_alpha()
        return surf

    def render(
        self,
        surface: pygame.surface.Surface,
        text: str,
        xy: tuple[int, int],
        size: int | None = None,
        color: tuple[int, int, int] | None = None,
        background: tuple[int, int, int] | None = None,
        force_update: bool = False,
    ) -> None:
        """Render given text, use stored data to render, if arguments change internal data and render."""
        surf = self.render_nosurf(text, size, color, background, force_update)

        if True in self.center:
            x, y = xy
            cx, cy = self.center
            w, h = surf.get_size()
            if cx:
                x -= w // 2
            if cy:
                y -= h // 2
            xy = (int(x), int(y))

        surface.blit(surf, xy)


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


class Object:
    """Object object."""

    __slots__ = (
        "Render_Priority",
        "game",
        "hidden",
        "id",
        "image",
        "location",
        "location_mode_on_resize",
        "name",
        "screen_size_last",
        "wh",
    )

    def __init__(self, name: str) -> None:
        """Set self.name to name, and other values for rendering.

        Defines the following attributes:
         self.name
         self.image
         self.location
         self.wh
         self.hidden
         self.location_mode_on_resize
         self.id
        """
        self.name = str(name)
        self.image: pygame.surface.Surface | None = None
        self.location = Vector2(
            round(SCREENSIZE[0] / 2),
            round(SCREENSIZE[1] / 2),
        )
        self.wh = 0, 0
        self.hidden = False
        self.location_mode_on_resize = "Scale"
        self.screen_size_last = SCREENSIZE

        self.id = 0
        self.game: Game
        self.Render_Priority: str | int

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}()"

    def get_image_zreo_no_fix(self) -> tuple[float, float]:
        """Return the screen location of the topleft point of self.image."""
        return (
            self.location[0] - self.wh[0] / 2,
            self.location[1] - self.wh[1] / 2,
        )

    def get_image_zero(self) -> tuple[int, int]:
        """Return the screen location of the topleft point of self.image fixed to integer values."""
        x, y = self.get_image_zreo_no_fix()
        return int(x), int(y)

    def get_rect(self) -> Rect:
        """Return a Rect object representing this Object's area."""
        return Rect(self.get_image_zero(), self.wh)

    def point_intersects(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> bool:
        """Return True if this Object intersects with a given screen location."""
        return self.get_rect().collidepoint(tuple(screen_location))

    def to_image_surface_location(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> tuple[int, int]:
        """Return the location a screen location would be at on the objects image. Can return invalid data."""
        # Get zero zero in image locations
        zx, zy = self.get_image_zero()  # Zero x and y
        sx, sy = screen_location  # Screen x and y
        return (
            int(sx) - zx,
            int(sy) - zy,
        )  # Location with respect to image dimensions

    def process(self, time_passed: float) -> None:
        """Process Object. Replace when calling this class."""

    def render(self, surface: pygame.surface.Surface) -> None:
        """Render self.image to surface if self.image is not None. Updates self.wh."""
        if self.image is None or self.hidden:
            return
        self.wh = self.image.get_size()
        x, y = self.get_image_zero()
        surface.blit(self.image, (int(x), int(y)))

    # pygame.draw.rect(surface, MAGENTA, self.get_rect(), 1)

    def __del__(self) -> None:
        """Delete self.image."""
        del self.image

    def screen_size_update(self) -> None:
        """Handle screensize is changes."""
        nx, ny = self.location

        if self.location_mode_on_resize == "Scale":
            ow, oh = self.screen_size_last
            nw, nh = SCREENSIZE

            x, y = self.location
            nx, ny = x * (nw / ow), y * (nh / oh)

        self.location = Vector2(nx, ny)
        self.screen_size_last = SCREENSIZE


class MultipartObject(Object, ObjectHandler):
    """Thing that is both an Object and an ObjectHandler, and is meant to be an Object made up of multiple Objects."""

    def __init__(self, name: str):
        """Initialize Object and ObjectHandler of self.

        Also set self._lastloc and self._lasthidden to None
        """
        Object.__init__(self, name)
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


class Tile(NamedTuple):
    """Represents a Tile."""

    color: int


class TileRenderer(Object):
    """Base class for all objects that need to render tiles."""

    __slots__ = ("back", "image_update", "tile_full", "tile_seperation")
    greyshift = GREYSHIFT
    tile_size = TILESIZE

    def __init__(
        self,
        name: str,
        game: Game,
        tile_seperation: int | None = None,
        background: tuple[int, int, int] | None = TILEDEFAULT,
    ) -> None:
        """Initialize renderer. Needs a game object for its cache and optional tile separation value and background RGB color.

        Defines the following attributes during initialization and uses throughout:
         self.game
         self.wh
         self.tile_seperation
         self.tile_full
         self.back
         and finally, self.image_update

        The following functions are also defined:
         self.clear_image
         self.render_tile
         self.update_image (but not implemented)
         self.process
        """
        super().__init__(name)
        self.game = game

        if tile_seperation is None:
            self.tile_seperation = self.tile_size / 3.75
        else:
            self.tile_seperation = tile_seperation

        self.tile_full = self.tile_size + self.tile_seperation
        self.back = background

        self.image_update = True

    def get_rect(self) -> Rect:
        """Return a Rect object representing this row's area."""
        wh = (
            self.wh[0] - self.tile_seperation * 2,
            self.wh[1] - self.tile_seperation * 2,
        )
        location = self.location[0] - wh[0] / 2, self.location[1] - wh[1] / 2
        return Rect(location, wh)

    def clear_image(self, tile_dimensions: tuple[int, int]) -> None:
        """Reset self.image using tile_dimensions tuple and fills with self.back. Also updates self.wh."""
        tw, th = tile_dimensions
        self.wh = (
            round(tw * self.tile_full + self.tile_seperation),
            round(th * self.tile_full + self.tile_seperation),
        )
        self.image = get_tile_container_image(self.wh, self.back)

    def render_tile(
        self,
        tile_object: Tile,
        tile_location: tuple[int, int],
    ) -> None:
        """Blit the surface of a given tile object onto self.image at given tile location. It is assumed that all tile locations are xy tuples."""
        x, y = tile_location
        surf = get_tile_image(tile_object, self.tile_size, self.greyshift)
        assert self.image is not None
        self.image.blit(
            surf,
            (
                round(x * self.tile_full + self.tile_seperation),
                round(y * self.tile_full + self.tile_seperation),
            ),
        )

    def update_image(self) -> None:
        """Process image changes, directed by self.image_update being True."""
        raise NotImplementedError

    def process(self, time_passed: float) -> None:
        """Call self.update_image() if self.image_update is True, then set self.update_image to False."""
        if self.image_update:
            self.update_image()
            self.image_update = False


class Cursor(TileRenderer):
    """Cursor Object."""

    __slots__ = ("holding_number_one", "tiles")
    greyshift = GREYSHIFT
    Render_Priority = "last"

    def __init__(self, game: Game) -> None:
        """Initialize cursor with a game it belongs to."""
        super().__init__("Cursor", game, background=None)

        self.holding_number_one = False
        self.tiles: deque[Tile] = deque()

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image((len(self.tiles), 1))

        for x in range(len(self.tiles)):
            self.render_tile(self.tiles[x], (x, 0))

    def is_pressed(self) -> bool:
        """Return True if the right mouse button is pressed."""
        return bool(pygame.mouse.get_pressed()[0])

    def get_held_count(self, count_number_one: bool = False) -> int:
        """Return the number of held tiles, can be discounting number one tile."""
        length = len(self.tiles)
        if self.holding_number_one and not count_number_one:
            return length - 1
        return length

    def is_holding(self, count_number_one: bool = False) -> bool:
        """Return True if the mouse is dragging something."""
        return self.get_held_count(count_number_one) > 0

    def get_held_info(
        self,
        count_number_one_tile: bool = False,
    ) -> tuple[Tile | None, int]:
        """Return color of tiles are and number of tiles held."""
        if not self.is_holding(count_number_one_tile):
            return None, 0
        return self.tiles[0], self.get_held_count(count_number_one_tile)

    def process(self, time_passed: float) -> None:
        """Process cursor."""
        x, y = pygame.mouse.get_pos()
        x = saturate(x, 0, SCREENSIZE[0])
        y = saturate(y, 0, SCREENSIZE[1])
        self.location = Vector2(x, y)
        if self.image_update:
            if len(self.tiles):
                self.update_image()
            else:
                self.image = None
            self.image_update = False

    def force_hold(self, tiles: Iterable[Tile]) -> None:
        """Pretty much it's drag but with no constraints."""
        for tile in tiles:
            if tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.image_update = True

    def drag(self, tiles: Iterable[Tile]) -> None:
        """Drag one or more tiles, as long as it's a list."""
        for tile in tiles:
            if tile is not None and tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.image_update = True

    def drop(
        self,
        number: int | None = None,
        allow_number_one_tile: bool = False,
    ) -> list[Tile]:
        """Return all of the tiles the Cursor is carrying."""
        if self.is_holding(allow_number_one_tile):
            if number is None:
                number = self.get_held_count(allow_number_one_tile)
            else:
                number = saturate(
                    number,
                    0,
                    self.get_held_count(allow_number_one_tile),
                )

            tiles = []
            for tile in (self.tiles.popleft() for i in range(number)):
                if tile.color == NUMBERONETILE and not allow_number_one_tile:
                    self.tiles.append(tile)
                    continue
                tiles.append(tile)
            self.image_update = True

            self.holding_number_one = NUMBERONETILE in {
                tile.color for tile in self.tiles
            }
            return tiles
        return []

    def drop_one_tile(self) -> Tile | None:
        """If holding the number one tile, drop it (returns it)."""
        if self.holding_number_one:
            not_number_one_tile = self.drop(None, False)
            one = self.drop(1, True)
            self.drag(not_number_one_tile)
            self.holding_number_one = False
            return one[0]
        return None


G = TypeVar("G", bound="Grid")


def gsc_bound_index(
    bounds_failure_return: T,
) -> Callable[
    [Callable[[G, tuple[int, int], *P], RT]],
    Callable[[G, tuple[int, int], *P], RT | T],
]:
    """Return a decorator for any grid or grid subclass that will keep index positions within bounds."""

    def gsc_bounds_keeper(
        function: Callable[[G, tuple[int, int], *P], RT],
    ) -> Callable[[G, tuple[int, int], *P], RT | T]:
        """Grid or Grid Subclass Decorator that keeps index positions within bounds, as long as index is first argument after self arg."""

        @wraps(function)
        def keep_within_bounds(
            self: G,
            index: tuple[int, int],
            *args: Unpack[P],
        ) -> RT | T:
            """Ensure a index position tuple is valid."""
            x, y = index
            if x < 0 or x >= self.size[0]:
                return bounds_failure_return
            if y < 0 or y >= self.size[1]:
                return bounds_failure_return
            return function(self, index, *args)

        return keep_within_bounds

    return gsc_bounds_keeper


class Grid(TileRenderer):
    """Grid object, used for boards and parts of other objects."""

    __slots__ = ("data", "size")

    def __init__(
        self,
        size: tuple[int, int],
        game: Game,
        tile_seperation: int | None = None,
        background: tuple[int, int, int] | None = TILEDEFAULT,
    ) -> None:
        """Grid Objects require a size and game at least."""
        super().__init__("Grid", game, tile_seperation, background)

        self.size = size

        self.data = array(
            [-6 for i in range(int(self.size[0] * self.size[1]))],
            int8,
        ).reshape(self.size)

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image(self.size)

        for y in range(self.size[1]):
            for x in range(self.size[0]):
                self.render_tile(Tile(self.data[y, x]), (x, y))

    def get_tile_point(
        self,
        screen_location: tuple[int, int] | Vector2,
    ) -> tuple[int, int] | None:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.point_intersects(screen_location):
            return None
        # Otherwise, find out where screen point is in image locations
        # board x and y
        bx, by = self.to_image_surface_location(screen_location)
        # Finally, return the full divides (no decimals) of xy location by self.tile_full.
        return int(bx // self.tile_full), int(by // self.tile_full)

    @gsc_bound_index(None)
    def place_tile(self, xy: tuple[int, int], tile: Tile) -> bool:
        """Place a Tile Object if permitted to do so. Return True if success."""
        x, y = xy
        if self.data[y, x] < 0:
            self.data[y, x] = tile.color
            del tile
            self.image_update = True
            return True
        return False

    @gsc_bound_index(None)
    def get_tile(self, xy: tuple[int, int], replace: int = -6) -> Tile | None:
        """Return a Tile Object from a given position in the grid if permitted. Return None on failure."""
        x, y = xy
        tile_color = int(self.data[y, x])
        if tile_color < 0:
            return None
        self.data[y, x] = replace
        self.image_update = True
        return Tile(tile_color)

    @gsc_bound_index(None)
    def get_info(self, xy: tuple[int, int]) -> Tile:
        """Return the Tile Object at a given position without deleting it from the Grid."""
        x, y = xy
        color = int(self.data[y, x])
        return Tile(color)

    def get_colors(self) -> list[int]:
        """Return a list of the colors of tiles within self."""
        colors = set()
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                info_color = int(self.data[y, x])
                assert info_color is not None
                colors.add(info_color)
        return list(colors)

    def is_empty(self, empty_color: int = -6) -> bool:
        """Return True if Grid is empty (all tiles are empty_color)."""
        colors = self.get_colors()
        # Colors should only be [-6] if empty
        return colors == [empty_color]

    def __del__(self) -> None:
        """Delete data."""
        super().__del__()
        del self.data


class Board(Grid):
    """Represents the board in the Game."""

    __slots__ = ("additions", "player", "variant_play", "wall_tiling")
    bcolor = ORANGE

    def __init__(self, player: Player, variant_play: bool = False) -> None:
        """Initialize player's board."""
        super().__init__((5, 5), player.game, background=self.bcolor)
        self.name = "Board"
        self.player = player

        self.variant_play = variant_play
        self.additions: dict[int, Tile | int | None] = {}

        self.wall_tiling = False

    def __repr__(self) -> str:
        """Return representation of self."""
        return (
            f"{self.__class__.__name__}({self.player!r}, {self.variant_play})"
        )

    def set_colors(self, keep_read: bool = True) -> None:
        """Reset tile colors."""
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if not keep_read or self.data[y, x] < 0:
                    self.data[y, x] = -(
                        (self.size[1] - y + x) % REGTILECOUNT + 1
                    )

    # print(self.data[y, x], end=' ')
    # print()
    # print('-'*10)

    def get_row(self, index: int) -> Generator[Tile, None, None]:
        """Return a row from self. Does not delete data from internal grid."""
        for x in range(self.size[0]):
            tile = self.get_info((x, index))
            assert tile is not None
            yield tile

    def get_column(self, index: int) -> Generator[Tile, None, None]:
        """Return a column from self. Does not delete data from internal grid."""
        for y in range(self.size[1]):
            tile = self.get_info((index, y))
            assert tile is not None
            yield tile

    def get_colors_in_row(
        self,
        index: int,
        exclude_negatives: bool = True,
    ) -> list[int]:
        """Return the colors placed in a given row in internal grid."""
        row_colors = [tile.color for tile in self.get_row(index)]
        if exclude_negatives:
            row_colors = [c for c in row_colors if c >= 0]
        ccolors = Counter(row_colors)
        return sorted(ccolors.keys())

    def get_colors_in_column(
        self,
        index: int,
        exclude_negatives: bool = True,
    ) -> list[int]:
        """Return the colors placed in a given row in internal grid."""
        column_colors = [tile.color for tile in self.get_column(index)]
        if exclude_negatives:
            column_colors = [c for c in column_colors if c >= 0]
        ccolors = Counter(column_colors)
        return sorted(ccolors.keys())

    def is_wall_tiling(self) -> bool:
        """Return True if in Wall Tiling Mode."""
        return self.wall_tiling

    def get_tile_for_cursor_by_row(self, row: int) -> Tile | None:
        """Return A COPY OF tile the mouse should hold. Returns None on failure."""
        if row in self.additions:
            data = self.additions[row]
            if isinstance(data, Tile):
                return data
        return None

    @gsc_bound_index(False)
    def can_place_tile_color_at_point(
        self,
        position: tuple[int, int],
        tile: Tile,
    ) -> bool:
        """Return True if tile's color is valid at given position."""
        column, row = position
        colors = set(
            self.get_colors_in_column(column) + self.get_colors_in_row(row),
        )
        return tile.color not in colors

    def get_rows_to_tile_map(self) -> dict[int, int]:
        """Return a dictionary of row numbers and row color to be wall tiled."""
        rows = {}
        for row, tile in self.additions.items():
            if not isinstance(tile, Tile):
                continue
            rows[row] = tile.color
        return rows

    def calculate_valid_locations_for_tile_row(
        self,
        row: int,
    ) -> tuple[int, ...]:
        """Return the valid drop columns of the additions tile for a given row."""
        valid = []
        # ??? Why overwriting row?
        if row in self.additions:
            tile = self.additions[row]
            if isinstance(tile, Tile):
                for column in range(self.size[0]):
                    if self.can_place_tile_color_at_point((column, row), tile):
                        valid.append(column)
                return tuple(valid)
        return ()

    def remove_invalid_additions(self) -> None:
        """Remove invalid additions that would not be placeable."""
        # In the wall-tiling phase, it may happen that you
        # are not able to move the rightmost tile of a certain
        # pattern line over to the wall because there is no valid
        # space left for it. In this case, you must immediately
        # place all tiles of that pattern line in your floor line.
        for row in range(self.size[1]):
            row_tile = self.additions[row]
            if not isinstance(row_tile, Tile):
                continue
            valid = self.calculate_valid_locations_for_tile_row(row)
            if not valid:
                floor = self.player.get_object_by_name("floor_line")
                assert isinstance(floor, FloorLine)
                floor.place_tile(row_tile)
                self.additions[row] = None

    @gsc_bound_index(False)
    def wall_tile_from_point(self, position: tuple[int, int]) -> bool:
        """Given a position, wall tile. Return success on placement. Also updates if in wall tiling mode."""
        success = False
        column, row = position
        at_point = self.get_info(position)
        assert at_point is not None
        if at_point.color <= 0 and row in self.additions:
            tile = self.additions[row]
            if isinstance(tile, Tile) and self.can_place_tile_color_at_point(
                position,
                tile,
            ):
                self.place_tile(position, tile)
                self.additions[row] = column
                # Update invalid placements after new placement
                self.remove_invalid_additions()
                success = True
        if not self.get_rows_to_tile_map():
            self.wall_tiling = False
        return success

    def wall_tiling_mode(self, moved_table: dict[int, Tile]) -> None:
        """Set self into Wall Tiling Mode. Finishes automatically if not in variant play mode."""
        self.wall_tiling = True
        for key, value in moved_table.items():
            key = int(key) - 1
            if key in self.additions:
                raise RuntimeError(
                    f"Key {key!r} Already in additions dictionary!",
                )
            self.additions[key] = value
        if not self.variant_play:
            for row in range(self.size[1]):
                if row in self.additions:
                    rowdata = [tile.color for tile in self.get_row(row)]
                    tile = self.additions[row]
                    if not isinstance(tile, Tile):
                        continue
                    negative_tile_color = -(tile.color + 1)
                    if negative_tile_color in rowdata:
                        column = rowdata.index(negative_tile_color)
                        self.place_tile((column, row), tile)
                        # Set data to the column placed in, use for scoring
                        self.additions[row] = column
                    else:
                        raise RuntimeError(
                            f"{negative_tile_color} not in row {row}!",
                        )
                else:
                    raise RuntimeError(f"{row} not in moved_table!")
            self.wall_tiling = False
        else:
            # Invalid additions can only happen in variant play mode.
            self.remove_invalid_additions()

    @gsc_bound_index(([], []))
    def get_touches_continuous(
        self,
        xy: tuple[int, int],
    ) -> tuple[list[Tile], list[Tile]]:
        """Return two lists, each of which contain all the tiles that touch the tile at given x y position, including that position."""
        rs, cs = self.size
        x, y = xy
        # Get row and column tile color data
        row = [tile.color for tile in self.get_row(y)]
        column = [tile.color for tile in self.get_column(x)]

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

        def get_all(lst: list[tuple[int, int]]) -> Generator[Tile, None, None]:
            """Return all of the self.get_info points for each value in lst."""
            for pos in lst:
                tile = self.get_info(pos)
                assert tile is not None
                yield tile

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
            real = (t.color >= 0 for t in self.get_row(row))
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
            real = (t.color >= 0 for t in self.get_column(column))
            if all(real):
                count += 1
        return count

    def get_filled_colors(self) -> int:
        """Return the number of completed colors on this board."""
        tiles = (
            self.get_info((x, y))
            for x in range(self.size[0])
            for y in range(self.size[1])
        )
        color_count = Counter(t.color for t in tiles if t is not None)
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

    def process(self, time_passed: float) -> None:
        """Process board."""
        if self.image_update and not self.variant_play:
            self.set_colors(True)
        super().process(time_passed)


class Row(TileRenderer):
    """Represents one of the five rows each player has."""

    __slots__ = ("color", "player", "size", "tiles")
    greyshift = GREYSHIFT

    def __init__(
        self,
        player: Player,
        size: int,
        tile_seperation: int | None = None,
        background: tuple[int, int, int] | None = None,
    ) -> None:
        """Initialize row."""
        super().__init__(
            "Row",
            player.game,
            tile_seperation,
            background,
        )
        self.player = player
        self.size = int(size)

        self.color = -6
        self.tiles = deque([Tile(self.color)] * self.size)

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}(%r, %i, ...)" % (
            self.game,
            self.size,
        )

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image((self.size, 1))

        for x in range(len(self.tiles)):
            self.render_tile(self.tiles[x], (x, 0))

    def get_tile_point(self, screen_location: tuple[int, int]) -> int | None:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        # `Grid.get_tile_point` inlined
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.point_intersects(screen_location):
            return None
        # Otherwise, find out where screen point is in image locations
        # board x and y
        bx, _by = self.to_image_surface_location(screen_location)
        # Finally, return the full divides (no decimals) of xy location by self.tile_full.

        return self.size - 1 - int(bx // self.tile_full)

    def get_placed(self) -> int:
        """Return the number of tiles in self that are not fake tiles, like grey ones."""
        return len([tile for tile in self.tiles if tile.color >= 0])

    def get_placeable(self) -> int:
        """Return the number of tiles permitted to be placed on self."""
        return self.size - self.get_placed()

    def is_full(self) -> bool:
        """Return True if this row is full."""
        return self.get_placed() == self.size

    def get_info(self, location: int) -> Tile | None:
        """Return tile at location without deleting it. Return None on invalid location."""
        index = self.size - 1 - location
        if index < 0 or index > len(self.tiles):
            return None
        return self.tiles[index]

    def can_place(self, tile: Tile) -> bool:
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

    def get_tile(self, replace: int = -6) -> Tile:
        """Return the leftmost tile while deleting it from self."""
        self.tiles.appendleft(Tile(replace))
        self.image_update = True
        return self.tiles.pop()

    def place_tile(self, tile: Tile) -> None:
        """Place a given Tile Object on self if permitted."""
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

    def can_place_tiles(self, tiles: list[Tile]) -> bool:
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

    def place_tiles(self, tiles: list[Tile]) -> None:
        """Place multiple tile objects on self if permitted."""
        if self.can_place_tiles(tiles):
            for tile in tiles:
                self.place_tile(tile)
        else:
            raise ValueError("Not allowed to place tiles.")

    def wall_tile(
        self,
        add_to_table: dict[str, list[Tile] | Tile | None],
        empty_color: int = -6,
    ) -> None:
        """Move tiles around and into add dictionary for the wall tiling phase of the game. Removes tiles from self."""
        if "tiles_for_box" not in add_to_table:
            add_to_table["tiles_for_box"] = []
        if not self.is_full():
            add_to_table[str(self.size)] = None
            return
        self.color = empty_color
        add_to_table[str(self.size)] = self.get_tile()
        for_box = add_to_table["tiles_for_box"]
        assert isinstance(for_box, list)
        for _i in range(self.size - 1):
            for_box.append(self.get_tile())

    def set_background(self, color: tuple[int, int, int] | None) -> None:
        """Set the background color for this row."""
        self.back = color
        self.image_update = True


class PatternLine(MultipartObject):
    """Represents multiple rows to make the pattern line."""

    __slots__ = ("player", "row_seperation")
    size = (5, 5)

    def __init__(self, player: Player, row_seperation: int = 0) -> None:
        """Initialize pattern line."""
        super().__init__("PatternLine")
        self.player = player
        self.row_seperation = row_seperation

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
        w = self.get_row(last - 1).wh[0]
        if w is None:
            raise RuntimeError(
                "Image Dimensions for Row Object (row.wh) are None!",
            )
        h1 = self.get_row(0).tile_full
        h = int(last * h1)
        self.wh = w, h
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

    def wall_tiling(self) -> dict[str, list[Tile] | Tile | None]:
        """Return a dictionary to be used with wall tiling. Removes tiles from rows."""
        values: dict[str, list[Tile] | Tile | None] = {}
        for rid in range(self.size[1]):
            self.get_row(rid).wall_tile(values)
        return values

    def process(self, time_passed: float) -> None:
        """Process all the rows that make up the pattern line."""
        if self.hidden != self._lasthidden:
            self.set_attr_all("image_update", True)
        super().process(time_passed)


class Text(Object):
    """Text object, used to render text with a given font."""

    __slots__ = ("_cxy", "_last", "font")

    def __init__(
        self,
        font_size: int,
        color: tuple[int, int, int],
        background: tuple[int, int, int] | None = None,
        cx: bool = True,
        cy: bool = True,
        name: str = "",
    ) -> None:
        """Initialize text."""
        super().__init__(f"Text{name}")
        self.font = Font(
            FONT,
            font_size,
            color,
            cx,
            cy,
            True,
            background,
            True,
        )
        self._cxy = cx, cy
        self._last: str | None = None

    def get_image_zero(self) -> tuple[int, int]:
        """Return the screen location of the topleft point of self.image."""
        x = int(self.location[0])
        y = int(self.location[1])
        if self._cxy[0]:
            x -= self.wh[0] // 2
        if self._cxy[1]:
            y -= self.wh[1] // 2
        return x, y

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"<{self.__class__.__name__} Object>"

    @staticmethod
    def get_font_height(font: str | Path, size: int) -> int:
        """Return the height of font at fontsize size."""
        return pygame.font.Font(font, size).get_height()

    def update_value(
        self,
        text: str | None,
        size: int | None = None,
        color: tuple[int, int, int] | None = None,
        background: tuple[int, int, int] | None = None,
    ) -> pygame.surface.Surface:
        """Return a surface of given text rendered in FONT."""
        self.image = self.font.render_nosurf(text, size, color, background)
        return self.image

    def get_surface(self) -> pygame.surface.Surface:
        """Return self.image."""
        assert self.image is not None
        return self.image

    def get_tile_point(self, location: tuple[int, int]) -> None:
        """Set get_tile_point attribute so that errors are not raised."""
        return

    def process(self, time_passed: float) -> None:
        """Process text."""
        if self.font.last_text != self._last:
            self.update_value(self.font.last_text)
            self._last = self.font.last_text


class FloorLine(Row):
    """Represents a player's floor line."""

    size = 7
    number_one_color = NUMBERONETILE

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
        assert self.wh is not None, "Should be impossible."
        w, h = self.wh
        for x in range(self.size):
            xy = round(
                x * self.tile_full + self.tile_seperation + sx - w / 2,
            ), round(
                self.tile_seperation + sy - h / 2,
            )
            self.text.update_value(str(self.numbers[x]))
            self.text.location = Vector2(*xy)
            self.text.render(surface)

    # self.font.render(surface, str(self.numbers[x]), xy)

    def place_tile(self, tile: Tile) -> None:
        """Place a given Tile Object on self if permitted."""
        self.tiles.insert(self.get_placed(), tile)

        if tile.color == self.number_one_color:
            self.has_number_one_tile = True

        box_lid = self.player.game.get_object_by_name("BoxLid")
        assert isinstance(box_lid, BoxLid)

        def handle_end(end: Tile) -> None:
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
        empty_color: int = -6,
    ) -> tuple[list[Tile], Tile | None]:
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
            self.tiles.append(Tile(empty_color))
        self.image_update = True
        return tiles, number_one_tile

    def can_place_tiles(self, tiles: list[Tile]) -> bool:
        """Return True."""
        return True


class Factory(Grid):
    """Represents a Factory."""

    size = (2, 2)
    color = WHITE
    outline = BLUE
    out_size = 0.1

    def __init__(self, game: Game, factory_id: int) -> None:
        """Initialize factory."""
        super().__init__(self.size, game, background=None)
        self.number = factory_id
        self.name = f"Factory{self.number}"

        self.radius = math.ceil(
            self.tile_full * self.size[0] * self.size[1] / 3 + 3,
        )

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}(%r, %i)" % (self.game, self.number)

    def add_circle(self, surface: pygame.surface.Surface) -> None:
        """Add circle to self.image."""
        # if f"FactoryCircle{self.radius}" not in self.game.cache:
        rad = math.ceil(self.radius)
        surf = set_alpha(pygame.surface.Surface((2 * rad, 2 * rad)), 1)
        pygame.draw.circle(surf, self.outline, (rad, rad), rad)
        pygame.draw.circle(
            surf,
            self.color,
            (rad, rad),
            math.ceil(rad * (1 - self.out_size)),
        )
        # self.game.cache[f"FactoryCircle{self.radius}"] = surf
        # surf = self.game.cache[f"FactoryCircle{self.radius}"].copy()
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

    def fill(self, tiles: list[Tile]) -> None:
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

    def grab(self) -> list[Tile]:
        """Return all tiles on this factory."""
        return [
            tile
            for tile in (
                self.get_tile((x, y), -6)
                for x in range(self.size[0])
                for y in range(self.size[1])
            )
            if tile is not None and tile.color != -6
        ]

    def grab_color(self, color: int) -> tuple[list[Tile], list[Tile]]:
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

    teach = 4

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
        degrees = 360 / self.count
        for i in range(self.count):
            radians = math.radians(degrees * i)
            self.objects[i].location = Vector2(
                math.sin(radians) * self.size + self.location[0],
                math.cos(radians) * self.size + self.location[1],
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

    def play_tiles_from_bag(self, empty_color: int = -6) -> None:
        """Divy up tiles to each factory from the bag."""
        # For every factory we have,
        for fid in range(self.count):
            # Draw tiles for the factory
            drawn = []
            for _i in range(self.teach):
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
                        drawn.append(Tile(empty_color))
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


class TableCenter(Grid):
    """Object that represents the center of the table."""

    size = (6, 6)
    first_tile_color = NUMBERONETILE

    def __init__(self, game: Game, has_number_one_tile: bool = True) -> None:
        """Initialize center of table."""
        super().__init__(self.size, game, background=None)
        self.game = game
        self.name = "TableCenter"

        self.number_one_tile_exists = False
        if has_number_one_tile:
            self.add_number_one_tile()

        self.next_position = (0, 0)

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.game!r})"

    def add_number_one_tile(self) -> None:
        """Add the number one tile to the internal grid."""
        if not self.number_one_tile_exists:
            x, y = self.size
            self.place_tile((x - 1, y - 1), Tile(self.first_tile_color))
            self.number_one_tile_exists = True

    def add_tile(self, tile: Tile) -> None:
        """Add a Tile Object to the Table Center Grid."""
        self.place_tile(self.next_position, tile)
        x, y = self.next_position
        x += 1
        y += int(x // self.size[0])
        x %= self.size[0]
        y %= self.size[1]
        self.next_position = (x, y)
        self.image_update = True

    def add_tiles(self, tiles: Iterable[Tile], sort: bool = True) -> None:
        """Add multiple Tile Objects to the Table Center Grid."""
        for tile in tiles:
            self.add_tile(tile)
        if sort and tiles:
            self.reorder_tiles()

    def reorder_tiles(self, replace: int = -6) -> None:
        """Re-organize tiles by Color."""
        full = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if self.number_one_tile_exists:
                    tile = self.get_info((x, y))
                    assert tile is not None
                    if tile.color == self.first_tile_color:
                        continue
                at = self.get_tile((x, y), replace)

                if at is not None:
                    full.append(at)
        sorted_tiles = sorted(full, key=sort_tiles)
        self.next_position = (0, 0)
        self.add_tiles(sorted_tiles, False)

    def pull_tiles(self, tile_color: int, replace: int = -6) -> list[Tile]:
        """Remove all of the tiles of tile_color from the Table Center Grid."""
        to_pull: list[tuple[int, int]] = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                info_tile = self.get_info((x, y))
                assert info_tile is not None
                if info_tile.color == tile_color:
                    to_pull.append((x, y))
                elif (
                    self.number_one_tile_exists
                    and info_tile.color == self.first_tile_color
                ):
                    to_pull.append((x, y))
                    self.number_one_tile_exists = False
        tiles = []
        for pos in to_pull:
            tile = self.get_tile(pos, replace)
            assert tile is not None
            tiles.append(tile)
        self.reorder_tiles(replace)
        return tiles

    def process(self, time_passed: float) -> None:
        """Process factories."""
        if self.hidden:
            super().process(time_passed)
            return
        cursor = self.game.get_object_by_name("Cursor")
        assert isinstance(cursor, Cursor)
        if (
            cursor.is_pressed()
            and not cursor.is_holding()
            and not self.is_empty()
            and self.point_intersects(cursor.location)
        ):
            point = self.get_tile_point(cursor.location)
            # Shouldn't return none anymore since we have point_intersects now.
            assert point is not None
            tile = self.get_info(point)
            assert isinstance(tile, Tile)
            color_at_point = tile.color
            if color_at_point >= 0 and color_at_point < 5:
                cursor.drag(self.pull_tiles(color_at_point))
        super().process(time_passed)


class Bag:
    """Represents the bag full of tiles."""

    __slots__ = (
        "percent_each",
        "tile_count",
        "tile_names",
        "tile_types",
        "tiles",
    )

    def __init__(self, tile_count: int = 100, tile_types: int = 5) -> None:
        """Initialize bag of tiles."""
        self.tile_count = int(tile_count)
        self.tile_types = int(tile_types)
        self.tile_names = [chr(65 + i) for i in range(self.tile_types)]
        self.percent_each = (self.tile_count / self.tile_types) / 100
        self.tiles: deque[str]
        self.full_reset()

    def full_reset(self) -> None:
        """Reset the bag to a full, re-randomized bag."""
        self.tiles = deque(
            gen_random_proper_seq(
                self.tile_count,
                **dict.fromkeys(self.tile_names, self.percent_each),
            ),
        )

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}(%i, %i)" % (
            self.tile_count,
            self.tile_types,
        )

    def reset(self) -> None:
        """Randomize all the tiles in the bag."""
        self.tiles = deque(randomize(self.tiles))

    def get_color(self, tile_name: str) -> int:
        """Return the color of a named tile."""
        if tile_name not in self.tile_names:
            raise ValueError(f"Tile Name {tile_name} Not Found!")
        return self.tile_names.index(tile_name)

    def get_tile(self, tile_name: str) -> Tile:
        """Return a Tile Object from a tile name."""
        return Tile(self.get_color(tile_name))

    def get_count(self) -> int:
        """Return number of tiles currently held."""
        return len(self.tiles)

    def is_empty(self) -> bool:
        """Return True if no tiles are currently held."""
        return self.get_count() == 0

    def draw_tile(self) -> Tile | None:
        """Return a random Tile Object from the bag. Return None if no tiles to draw."""
        if not self.is_empty():
            return self.get_tile(self.tiles.pop())
        return None

    def get_name(self, tile_color: int) -> str:
        """Return the name of a tile given it's color."""
        try:
            return self.tile_names[tile_color]
        except IndexError as exc:
            raise ValueError("Invalid Tile Color!") from exc

    def add_tile(self, tile_object: Tile) -> None:
        """Add a Tile Object to the bag."""
        name = self.get_name(int(tile_object.color))
        range_ = (0, len(self.tiles) - 1)
        if range_[1] - range_[0] <= 1:
            index = 0
        else:
            # S311 Standard pseudo-random generators are not suitable for cryptographic purposes
            index = random.randint(range_[0], range_[1])  # noqa: S311
        # self.tiles.insert(random.randint(0, len(self.tiles)-1), self.get_name(int(tile_object.color)))
        self.tiles.insert(index, name)
        del tile_object

    def add_tiles(self, tile_objects: Iterable[Tile]) -> None:
        """Add multiple Tile Objects to the bag."""
        for tile_object in tile_objects:
            self.add_tile(tile_object)


class BoxLid(Object):
    """BoxLid Object, represents the box lid were tiles go before being added to the bag again."""

    def __init__(self, game: Game) -> None:
        """Initialize box lid."""
        super().__init__("BoxLid")
        self.game = game
        self.tiles: deque[Tile] = deque()

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.game!r})"

    def add_tile(self, tile: Tile) -> None:
        """Add a tile to self."""
        if tile.color >= 0 and tile.color < 5:
            self.tiles.append(tile)
            return
        raise ValueError(
            f"BoxLid.add_tile tried to add an invalid tile to self ({tile.color = }).",
        )

    def add_tiles(self, tiles: Iterable[Tile]) -> None:
        """Add multiple tiles to self."""
        for tile in tiles:
            self.add_tile(tile)

    def get_tiles(self) -> list[Tile]:
        """Return all tiles in self while deleting them from self."""
        return [self.tiles.popleft() for i in range(len(self.tiles))]

    def is_empty(self) -> bool:
        """Return True if self is empty (no tiles on it)."""
        return len(self.tiles) == 0


class Player(MultipartObject):
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

        self.add_object(Board(self, self.varient_play))
        self.add_object(PatternLine(self))
        self.add_object(FloorLine(self))
        self.add_object(Text(SCOREFONTSIZE, SCORECOLOR))

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

    def update_score(self) -> None:
        """Update the scorebox for this player."""
        score_box = self.get_object_by_name("Text")
        assert isinstance(score_box, Text)
        score_box.update_value(f"Player {self.player_id + 1}: {self.score}")

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

    def end_of_game_trigger(self) -> None:
        """Handle end of game.

        Called by end state when game is over
        Hide pattern lines and floor line.
        """
        pattern = self.get_object_by_name("PatternLine")
        floor = self.get_object_by_name("floor_line")

        pattern.hidden = True
        floor.hidden = True

    def reset_position(self) -> None:
        """Reset positions of all parts of self based off self.location."""
        x, y = self.location

        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)
        bw, bh = board.wh
        board.location = Vector2(x + bw // 2, y)

        pattern_line = self.get_object_by_name("PatternLine")
        assert isinstance(pattern_line, PatternLine)
        lw = pattern_line.wh[0] // 2
        pattern_line.location = Vector2(x - lw, y)

        floor_line = self.get_object_by_name("floor_line")
        assert isinstance(floor_line, FloorLine)
        floor_line.wh[0]
        floor_line.location = Vector2(
            int(x - lw * (2 / 3) + TILESIZE / 3.75),
            int(y + bh * (2 / 3)),
        )

        text = self.get_object_by_name("Text")
        assert isinstance(text, Text)
        text.location = Vector2(x - (bw // 3), y - (bh * 2 // 3))

    def wall_tiling(self) -> None:
        """Do the wall tiling phase of the game for this player."""
        self.is_wall_tiling = True
        pattern_line = self.get_object_by_name("PatternLine")
        assert isinstance(pattern_line, PatternLine)
        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)
        box_lid = self.game.get_object_by_name("BoxLid")
        assert isinstance(box_lid, BoxLid)

        data = pattern_line.wall_tiling()
        tiles_for_box = data["tiles_for_box"]
        assert isinstance(tiles_for_box, list)
        box_lid.add_tiles(tiles_for_box)
        del data["tiles_for_box"]

        cleaned = {}
        for key, value in data.items():
            if not isinstance(value, Tile):
                continue
            cleaned[int(key)] = value

        board.wall_tiling_mode(cleaned)

    def done_wall_tiling(self) -> bool:
        """Return True if internal Board is done wall tiling."""
        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)
        return not board.is_wall_tiling()

    def next_round(self) -> None:
        """Handle end of wall tiling."""
        self.is_wall_tiling = False

    def score_phase(self) -> Tile | None:
        """Do the scoring phase of the game for this player. Return number one tile or None."""
        board = self.get_object_by_name("Board")
        floor_line = self.get_object_by_name("floor_line")
        box_lid = self.game.get_object_by_name("BoxLid")
        assert isinstance(board, Board)
        assert isinstance(floor_line, FloorLine)
        assert isinstance(box_lid, BoxLid)

        def saturatescore() -> None:
            if self.score < 0:
                self.score = 0

        self.score += board.score_additions()
        self.score += floor_line.score_tiles()
        saturatescore()

        tiles_for_box, number_one = floor_line.get_tiles()
        box_lid.add_tiles(tiles_for_box)

        self.update_score()

        return number_one

    def end_of_game_scoring(self) -> None:
        """Update final score with additional end of game points."""
        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)

        self.score += board.end_of_game_scoreing()

        self.update_score()

    def has_horzontal_line(self) -> bool:
        """Return True if this player has a horizontal line on their game board filled."""
        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)

        return board.has_filled_row()

    def get_horizontal_lines(self) -> int:
        """Return the number of filled horizontal lines this player has on their game board."""
        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)

        return board.get_filled_rows()

    def process(self, time_passed: float) -> None:
        """Process Player."""
        if not self.is_turn:  # Is our turn?
            self.set_attr_all("hidden", self.hidden)
            super().process(time_passed)
            return
        if self.hidden and self.is_wall_tiling and self.varient_play:
            # If hidden, not anymore. Our turn.
            self.hidden = False
        if self.networked:  # We are networked.
            self.set_attr_all("hidden", self.hidden)
            super().process(time_passed)
            return

        cursor = self.game.get_object_by_name("Cursor")
        assert isinstance(cursor, Cursor)
        box_lid = self.game.get_object_by_name("BoxLid")
        assert isinstance(box_lid, BoxLid)
        pattern_line = self.get_object_by_name("PatternLine")
        assert isinstance(pattern_line, PatternLine)
        floor_line = self.get_object_by_name("floor_line")
        assert isinstance(floor_line, FloorLine)
        board = self.get_object_by_name("Board")
        assert isinstance(board, Board)

        if not cursor.is_pressed():
            # Mouse up
            if self.just_held:
                self.just_held = False
            if self.just_dropped:
                self.just_dropped = False
            self.set_attr_all("hidden", self.hidden)
            super().process(time_passed)
            return

        # Mouse down
        obj, point = self.get_intersection(cursor.location)
        if obj is None or point is None:
            if self.is_wall_tiling and self.done_wall_tiling():
                self.next_round()
                self.game.next_turn()
            self.set_attr_all("hidden", self.hidden)
            super().process(time_passed)
            return
        # Something pressed
        if cursor.is_holding():  # Cursor holding tiles
            move_made = False
            if not self.is_wall_tiling:  # Is wall tiling:
                if obj == "PatternLine":
                    pos, row_number = point
                    row = pattern_line.get_row(row_number)
                    if not row.is_full():
                        info = row.get_info(pos)
                        if info is not None and info.color < 0:
                            _color, _held = cursor.get_held_info()
                            todrop = min(
                                pos + 1,
                                row.get_placeable(),
                            )
                            tiles = cursor.drop(todrop)
                            if row.can_place_tiles(tiles):
                                row.place_tiles(tiles)
                                move_made = True
                            else:
                                cursor.force_hold(tiles)
                elif obj == "floor_line":
                    tiles_to_add = cursor.drop()
                    if floor_line.is_full():
                        # Floor is full,
                        # Add tiles to box instead.
                        box_lid.add_tiles(tiles_to_add)
                    elif floor_line.get_placeable() < len(
                        tiles_to_add,
                    ):
                        # Floor is not full but cannot fit all in floor line.
                        # Add tiles to floor line and then to box
                        while len(tiles_to_add) > 0:
                            if floor_line.get_placeable() > 0:
                                floor_line.place_tile(
                                    tiles_to_add.pop(),
                                )
                            else:
                                box_lid.add_tile(
                                    tiles_to_add.pop(),
                                )
                    else:
                        # Otherwise add to floor line for all.
                        floor_line.place_tiles(tiles_to_add)
                    move_made = True
            elif not self.just_held and obj == "Board":
                tile = board.get_info(point)
                assert isinstance(tile, Tile)
                if tile.color == -6:
                    # Cursor holding and wall tiling
                    _column, row_id = point
                    cursor_tile = cursor.drop(1)[0]
                    board_tile = board.get_tile_for_cursor_by_row(
                        row_id,
                    )
                    if (
                        board_tile is not None
                        and cursor_tile.color == board_tile.color
                        and board.wall_tile_from_point(point)
                    ):
                        self.just_dropped = True
                        pattern_line.get_row(
                            row_id,
                        ).set_background(None)
            if move_made and not self.is_wall_tiling:
                if cursor.holding_number_one:
                    one_tile = cursor.drop_one_tile()
                    assert one_tile is not None
                    floor_line.place_tile(one_tile)
                if cursor.get_held_count(True) == 0:
                    self.game.next_turn()
        elif self.is_wall_tiling and obj == "Board" and not self.just_dropped:
            # Mouse down, something pressed, and not holding anything
            # Wall tiling, pressed, not holding
            _column_number, row_number = point
            tile = board.get_tile_for_cursor_by_row(
                row_number,
            )
            if tile is not None:
                cursor.drag([tile])
                self.just_held = True
        if self.is_wall_tiling and self.done_wall_tiling():
            self.next_round()
            self.game.next_turn()
        self.set_attr_all("hidden", self.hidden)
        super().process(time_passed)


class Button(Text):
    """Button Object."""

    textcolor = BUTTONTEXTCOLOR
    backcolor = BUTTONBACKCOLOR

    def __init__(
        self,
        state: MenuState,
        name: str,
        minimum_size: int = 10,
        initial_value: str = "",
        font_size: int = BUTTONFONTSIZE,
    ) -> None:
        """Initialize button."""
        super().__init__(font_size, self.textcolor, background=None)
        self.name = f"Button{name}"
        self.state = state

        self.minsize = int(minimum_size)
        self.update_value(initial_value)

        self.borderWidth = math.floor(font_size / 12)  # 5

        self.delay = 0.6
        self.cur_time = 1.0

        self.action: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"Button({self.name[6:]}, {self.state}, {self.minsize}, {self.font.last_text}, {self.font.pyfont})"

    def get_height(self) -> int:
        """Return font height."""
        return self.font.get_height()

    def bind_action(self, function: Callable[[], None]) -> None:
        """When self is pressed, call given function exactly once. Function takes no arguments."""
        self.action = function

    def update_value(
        self,
        text: str | None,
        size: int | None = None,
        color: tuple[int, int, int] | None = None,
        background: tuple[int, int, int] | None = None,
    ) -> pygame.surface.Surface:
        """Update button text."""
        disp = str(text or "").center(self.minsize)
        surface = super().update_value(f" {disp} ", size, color, background)
        self.font.last_text = disp
        return surface

    def render(self, surface: pygame.surface.Surface) -> None:
        """Render button."""
        if not self.hidden:
            text_rect = self.get_rect()
            # if PYGAME_VERSION < 201:
            # pygame.draw.rect(surface, self.backcolor, text_rect)
            # pygame.draw.rect(surface, BLACK, text_rect, self.borderWidth)
            # else:
            pygame.draw.rect(
                surface,
                self.backcolor,
                text_rect,
                border_radius=20,
            )
            pygame.draw.rect(
                surface,
                BLACK,
                text_rect,
                width=self.borderWidth,
                border_radius=20,
            )
        super().render(surface)

    def is_pressed(self) -> bool:
        """Return True if this button is pressed."""
        assert self.state.game is not None
        cursor = self.state.game.get_object_by_name("Cursor")
        assert isinstance(cursor, Cursor)
        return (
            not self.hidden
            and cursor.is_pressed()
            and self.point_intersects(cursor.location)
        )

    def process(self, time_passed: float) -> None:
        """Call self.action one time when pressed, then wait self.delay to call again."""
        if self.cur_time > 0:
            self.cur_time = max(self.cur_time - time_passed, 0)
        elif self.is_pressed():
            self.action()
            self.cur_time = self.delay
        if self.font.last_text != self._last:
            self.textSize = self.font.pyfont.size(f" {self.font.last_text} ")
        super().process(time_passed)


class GameState:
    """Base class for all game states."""

    __slots__ = ("game", "name")

    def __init__(self, name: str) -> None:
        """Initialize state with a name, set self.game to None to be overwritten later."""
        self.game: Game | None = None
        self.name = name

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"<{self.__class__.__name__} {self.name}>"

    def entry_actions(self) -> None:
        """Perform entry actions for this GameState."""

    def do_actions(self) -> None:
        """Perform actions for this GameState."""

    def check_state(self) -> str | None:
        """Check state and return new state. None remains in current state."""
        return None

    def exit_actions(self) -> None:
        """Perform exit actions for this GameState."""


class MenuState(GameState):
    """Game State where there is a menu with buttons."""

    button_minimum = 10
    fontsize = BUTTONFONTSIZE

    def __init__(self, name: str) -> None:
        """Initialize GameState and set up self.bh."""
        super().__init__(name)
        self.bh = Text.get_font_height(FONT, self.fontsize)

        self.next_state: str | None = None

    def add_button(
        self,
        name: str,
        value: str,
        action: Callable[[], None],
        location: tuple[int, int] | None = None,
        size: int = fontsize,
        minlen: int = button_minimum,
    ) -> int:
        """Add a new Button object to self.game with arguments. Return button id."""
        button = Button(self, name, minlen, value, size)
        button.bind_action(action)
        if location is not None:
            button.location = Vector2(*location)
        assert self.game is not None
        self.game.add_object(button)
        return button.id

    def add_text(
        self,
        name: str,
        value: str,
        location: tuple[int, int],
        color: tuple[int, int, int] = BUTTONTEXTCOLOR,
        cx: bool = True,
        cy: bool = True,
        size: int = fontsize,
    ) -> int:
        """Add a new Text object to self.game with arguments. Return text id."""
        text = Text(size, color, None, cx, cy, name)
        text.location = Vector2(*location)
        text.update_value(value)
        assert self.game is not None
        self.game.add_object(text)
        return text.id

    def entry_actions(self) -> None:
        """Clear all objects, add cursor object, and set up next_state."""
        self.next_state = None

        assert self.game is not None
        self.game.rm_star()
        self.game.add_object(Cursor(self.game))

    def set_var(self, attribute: str, value: object) -> None:
        """Set MenuState.{attribute} to {value}."""
        setattr(self, attribute, value)

    def to_state(self, state_name: str) -> Callable[[], None]:
        """Return a function that will change game state to state_name."""

        def to_state_name() -> None:
            """Set MenuState.next_state to {state_name}."""
            self.next_state = state_name

        return to_state_name

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

    def check_state(self) -> str | None:
        """Return self.next_state."""
        return self.next_state


class InitState(GameState):
    """Initialize state."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize self."""
        super().__init__("Init")

    def entry_actions(self) -> None:
        """Register keyboard handlers."""
        assert self.game is not None
        assert self.game.keyboard is not None
        self.game.keyboard.add_listener("\x7f", "Delete")
        self.game.keyboard.bind_action("Delete", "screenshot", 5)

        self.game.keyboard.add_listener("\x1b", "Escape")
        self.game.keyboard.bind_action("Escape", "raise_close", 5)

        self.game.keyboard.add_listener("0", "Debug")
        self.game.keyboard.bind_action("Debug", "debug", 5)

    def check_state(self) -> str:
        """Go to title state."""
        return "Title"


class TitleScreen(MenuState):
    """Game state when the title screen is up."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize title."""
        super().__init__("Title")

    def entry_actions(self) -> None:
        """Set up buttons."""
        super().entry_actions()
        sw, sh = SCREENSIZE
        self.add_button(
            "ToSettings",
            "New Game",
            self.to_state("Settings"),
            (sw // 2, sh // 2 - self.bh // 2),
        )
        self.add_button(
            "ToCredits",
            "Credits",
            self.to_state("Credits"),
            (sw // 2, sh // 2 + self.bh * 3),
            int(self.fontsize / 1.5),
        )
        assert self.game is not None
        self.add_button(
            "Quit",
            "Quit",
            self.game.raise_close,
            (sw // 2, sh // 2 + self.bh * 4),
            int(self.fontsize / 1.5),
        )


class CreditsScreen(MenuState):
    """Game state when credits for original game are up."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize credits."""
        super().__init__("Credits")

    def check_state(self) -> str:
        """Return to title."""
        return "Title"


class SettingsScreen(MenuState):
    """Game state when user is defining game type, players, etc."""

    def __init__(self) -> None:
        """Initialize settings."""
        super().__init__("Settings")

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

        sw, sh = SCREENSIZE
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

        # TEMPORARY: Hide everything to do with "Host Mode", networked games aren't done yet.
        assert self.game is not None
        self.game.set_attr_all("hidden", True)

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


class PhaseWallTilingNetworked(PhaseWallTiling):
    """Wall tiling networked state."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize will tiling networked."""
        GameState.__init__(self, "WallTilingNetworked")

    def check_state(self) -> str:
        """Go to networked next prepare."""
        return "PrepareNextNetworked"


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


class PhasePrepareNextNetworked(PhasePrepareNext):
    """Prepare for next, networked."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize prepare for next stage."""
        GameState.__init__(self, "PrepareNextNetworked")

    def check_state(self) -> str:
        """Go to networked end."""
        return "EndNetworked"


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
            (SCREENSIZE[0] // 2, SCREENSIZE[1] * 4 // 5),
        )
        buttontitle = self.game.get_object(bid)
        assert isinstance(buttontitle, Button)
        buttontitle.Render_Priority = "last-1"
        buttontitle.cur_time = 2

        # Add score board
        x = SCREENSIZE[0] // 2
        y = 10
        for idx, line in enumerate(self.wininf.split("\n")):
            self.add_text(f"Line{idx}", line, (x, y), cx=True, cy=False)
            # self.game.get_object(bid).Render_Priority = f'last{-(2+idx)}'
            button = self.game.get_object(bid)
            assert isinstance(button, Button)
            button.Render_Priority = "last-2"
            y += self.bh


class EndScreenNetworked(EndScreen):
    """Networked end screen."""

    def __init__(self) -> None:
        """Initialize end screen."""
        MenuState.__init__(self, "EndNetworked")
        self.ranking = {}
        self.wininf = ""

    def check_state(self) -> str:
        """Go to title."""
        return "Title"


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
                InitState(),
                TitleScreen(),
                CreditsScreen(),
                SettingsScreen(),
                PhaseFactoryOffer(),
                PhaseWallTiling(),
                PhasePrepareNext(),
                EndScreen(),
                PhaseFactoryOfferNetworked(),
                PhaseWallTilingNetworked(),
                PhasePrepareNextNetworked(),
                EndScreenNetworked(),
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

    def debug(self) -> None:
        """Debug."""

    def screenshot(self) -> None:
        """Save a screenshot of this game's most recent frame."""
        surface = pygame.surface.Surface(SCREENSIZE)
        self.render(surface)
        str_time = "-".join(time.asctime().split(" "))
        filename = f"Screenshot_at_{str_time}.png"

        if not os.path.exists("Screenshots"):
            os.mkdir("Screenshots")

        surface.unlock()
        pygame.image.save(
            surface,
            os.path.join("Screenshots", filename),
            filename,
        )
        del surface

        savepath = os.path.join(os.getcwd(), "Screenshots")

        print(f'Saved screenshot as "{filename}" in "{savepath}".')

    def raise_close(self) -> None:
        """Raise a window close event."""
        pygame.event.post(pygame.event.Event(QUIT))

    def add_states(self, states: Iterable[GameState]) -> None:
        """Add game states to self."""
        for state in states:
            if not isinstance(state, GameState):
                raise ValueError(
                    f'"{state}" Object is not a subclass of GameState!',
                )
            state.game = self
            self.states[state.name] = state

    def set_state(self, new_state_name: str) -> None:
        """Change states and perform any exit / entry actions."""
        # Ensure the new state is valid.
        if new_state_name not in self.states:
            raise ValueError(f'State "{new_state_name}" does not exist!')

        # If we have an active state,
        if self.active_state is not None:
            # Perform exit actions
            self.active_state.exit_actions()

        # The active state is the new state
        self.active_state = self.states[new_state_name]
        # Perform entry actions for new active state
        self.active_state.entry_actions()

    def update_state(self) -> None:
        """Perform the actions of the active state and potentially change states."""
        # Only continue if there is an active state
        if self.active_state is None:
            return

        # Perform the actions of the active state and check conditions
        self.active_state.do_actions()

        new_state_name = self.active_state.check_state()
        if new_state_name is not None:
            self.set_state(new_state_name)

    def add_object(self, obj: Object) -> None:
        """Add an object to the game."""
        obj.game = self
        super().add_object(obj)

    def render(self, surface: pygame.surface.Surface) -> None:
        """Render all of self.objects to the screen."""
        surface.fill(self.background_color)
        self.render_objects(surface)

    def process(self, time_passed: float) -> None:
        """Process all the objects and self."""
        if not self.initialized_state and self.keyboard is not None:
            self.set_state("Init")
            self.initialized_state = True
        self.process_objects(time_passed)
        self.update_state()

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

        cx, cy = SCREENSIZE[0] / 2, SCREENSIZE[1] / 2
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


class Keyboard:
    """Keyboard object, handles keyboard input."""

    __slots__ = ("actions", "active", "delay", "keys", "target", "time")

    def __init__(
        self,
        target: Game,
        **kwargs: tuple[str, str],
    ) -> None:
        """Initialize keyboard."""
        self.target = target
        self.target.keyboard = self

        # Map of keyboard events to names
        self.keys: dict[str, str] = {}
        # Map of keyboard event names to functions
        self.actions: dict[str, Callable[[], None]] = {}
        # Map of names to time until function should be called again
        self.time: dict[str, float] = {}
        # Map of names to duration timer waits for function recalls
        self.delay: dict[str, float | None] = {}
        # Map of names to boolian of pressed or not
        self.active: dict[str, bool] = {}

        for name in kwargs:
            if not hasattr(kwargs[name], "__iter__"):
                raise ValueError(
                    "Keyword arguments must be given as name=[key, self.target.function_name, delay]",
                )
            # if len(kwargs[name]) == 2:
            key, function_name = kwargs[name]
            # elif len(kwargs[name]) == 3:
            # key, function_name, _delay = kwargs[name]
            # else:
            # raise ValueError
            self.add_listener(key, name)
            self.bind_action(name, function_name)

    def __repr__(self) -> str:
        """Return representation of self."""
        return f"{self.__class__.__name__}({self.target!r})"

    def is_pressed(self, key: str) -> bool:
        """Return True if <key> is pressed."""
        return self.active.get(key, False)

    def add_listener(self, key: str, name: str) -> None:
        """Listen for key down events with event.key == key argument and when that happens set self.actions[name] to true."""
        self.keys[key] = name  # key to name
        self.actions[name] = lambda: None  # name to function
        self.time[name] = 0  # name to time until function recall
        self.delay[name] = None  # name to function recall delay
        self.active[name] = False  # name to boolian of pressed

    def get_function_from_target(
        self,
        function_name: str,
    ) -> Callable[[], None]:
        """Return function with name function_name from self.target."""
        if hasattr(self.target, function_name):
            attribute = getattr(self.target, function_name)
            assert callable(attribute)
            return cast("Callable[[], None]", attribute)
        return lambda: None

    def bind_action(
        self,
        name: str,
        target_function_name: str,
        delay: float | None = None,
    ) -> None:
        """Bind an event we are listening for to calling a function, can call multiple times if delay is not None."""
        self.actions[name] = self.get_function_from_target(
            target_function_name,
        )
        self.delay[name] = delay

    def set_active(self, name: str, value: bool) -> None:
        """Set active value for key name <name> to <value>."""
        if name in self.active:
            self.active[name] = bool(value)
            if not value:
                self.time[name] = 0

    def set_key(self, key: str, value: bool) -> None:
        """Set active value for key <key> to <value>."""
        if key in self.keys:
            self.set_active(self.keys[key], value)

    # elif isinstance(key, int) and key < 0x110000:
    # self.set_key(chr(key), value)

    def read_event(self, event: pygame.event.Event) -> None:
        """Handle an event."""
        if event.type == KEYDOWN:
            self.set_key(event.key, True)
        elif event.type == KEYUP:
            self.set_key(event.key, False)

    def read_events(self, events: Iterable[pygame.event.Event]) -> None:
        """Handle a list of events."""
        for event in events:
            self.read_event(event)

    def process(self, time_passed: float) -> None:
        """Send commands to self.target based on pressed keys and time."""
        for name in self.active:
            if self.active[name]:
                self.time[name] = max(self.time[name] - time_passed, 0)
                if self.time[name] == 0:
                    self.actions[name]()
                    delay = self.delay[name]
                    if delay is not None:
                        self.time[name] = delay
                    else:
                        self.time[name] = math.inf


def network_shutdown() -> None:
    """Handle network shutdown."""


def run() -> None:
    """Run program."""
    # global game
    global SCREENSIZE
    # Set up the screen
    screen = pygame.display.set_mode(SCREENSIZE, RESIZABLE, 16)
    pygame.display.set_caption(f"{__title__} {__version__}")
    # pygame.display.set_icon(pygame.image.load('icon.png'))
    pygame.display.set_icon(get_tile_image(Tile(5), 32))

    # Set up the FPS clock
    clock = pygame.time.Clock()

    game = Game()
    keyboard = Keyboard(game)

    music_end = USEREVENT + 1  # This event is sent when a music track ends

    # Set music end event to our new event
    pygame.mixer.music.set_endevent(music_end)

    # Load and start playing the music
    # pygame.mixer.music.load('sound/')
    # pygame.mixer.music.play()

    running = True

    # While the game is active
    while running:
        # Event handler
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == music_end:
                # If the music ends, stop it and play it again.
                pygame.mixer.music.stop()
                pygame.mixer.music.play()
            elif event.type == VIDEORESIZE:
                SCREENSIZE = event.size
                game.screen_size_update()
            else:
                # If it's not a quit or music end event, tell the keyboard handler about it.
                keyboard.read_event(event)

        # Get the time passed from the FPS clock
        time_passed = clock.tick(FPS)
        time_passed_secconds = time_passed / 1000

        # Process the game
        game.process(time_passed_secconds)
        keyboard.process(time_passed_secconds)

        # Render the grid to the screen.
        game.render(screen)

        # Update the display
        pygame.display.update()
    # Once the game has ended, stop the music and de-initalize pygame.
    pygame.mixer.music.stop()


def save_crash_img() -> None:
    """Save the last frame before the game crashed."""
    surface = pygame.display.get_surface().copy()
    str_time = "-".join(time.asctime().split(" "))
    filename = f"Crash_at_{str_time}.png"

    if not os.path.exists("Screenshots"):
        os.mkdir("Screenshots")

    # surface.lock()
    pygame.image.save(surface, os.path.join("Screenshots", filename), filename)
    # surface.unlock()
    del surface

    savepath = os.path.join(os.getcwd(), "Screenshots")

    print(f'Saved screenshot as "{filename}" in "{savepath}".')


def cli_run() -> None:
    """Run from command line interface."""
    # Linebreak before, as pygame prints a message on import.
    print(f"\n{__title__} v{__version__}\nProgrammed by {__author__}.")
    try:
        # Initialize Pygame
        _success, fail = pygame.init()
        if fail > 0:
            print(
                "Warning! Some modules of Pygame have not initialized properly!",
            )
            print(
                "This can occur when not all required modules of SDL, which pygame utilizes, are installed.",
            )
        run()
    # except BaseException as ex:
    # reraise = True#False
    ##
    # print('Debug: Activating Post motem.')
    # import pdb
    # pdb.post_mortem()
    ##
    # try:
    # save_crash_img()
    # except BaseException as svex:
    # print(f'Could not save crash screenshot: {", ".join(svex.args)}')
    # try:
    # import errorbox
    # except ImportError:
    # reraise = True
    # print(f'A {type(ex).__name__} Error Has Occored: {", ".join(ex.args)}')
    # else:
    # errorbox.errorbox('Error', f'A {type(ex).__name__} Error Has Occored: {", ".join(ex.args)}')
    # if reraise:
    # raise
    finally:
        pygame.quit()
        network_shutdown()


if __name__ == "__main__":
    cli_run()
