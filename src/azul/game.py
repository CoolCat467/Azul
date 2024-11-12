"""Azul board game clone, now on the computer!"""

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
import os
import random
import time
from collections import Counter, deque
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple

import pygame
from numpy import array
from pygame.locals import (
    KEYDOWN,
    KEYUP,
    QUIT,
    RESIZABLE,
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
from azul.Vector2 import Vector2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

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
    clear: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> pygame.surface.Surface:
    """Remove unneccicary pixels from image."""
    surface = surface.convert_alpha()
    w, h = surface.get_size()
    surface.lock()

    def find_end(
        iterfunc: Callable[[int], Iterable[tuple[int, int, int, int]]],
        rangeobj: Iterable[int],
    ) -> int:
        for x in rangeobj:
            if not all(y == clear for y in iterfunc(x)):
                return x
        return x

    def column(x: int) -> tuple[int, int, int, int]:
        return (surface.get_at((x, y)) for y in range(h))

    def row(y: int) -> tuple[int, int, int, int]:
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
) -> (
    tuple[float, float, float]
    | tuple[int, int, int]
    | tuple[tuple[int, int, int], tuple[int, int, int]]
    | None
):
    """Return the color a given tile should be."""
    if tile_color < 0:
        if tile_color == -6:
            return GREY
        color = tile_colors[abs(tile_color + 1)]
        assert len(color) == 3
        return lerp_color(color, GREY, greyshift)
    if tile_color < 5:
        return tile_colors[tile_color]
    if tile_color >= 5:
        raise ValueError(
            "Cannot properly return tile colors greater than five!",
        )
    return None


@lru_cache
def get_tile_symbol_and_color(
    tile_color: int,
    greyshift: float = GREYSHIFT,
) -> tuple[str, tuple[int, int, int]] | None:
    """Return the color a given tile should be."""
    if tile_color < 0:
        if tile_color == -6:
            return " ", GREY
        symbol, scolor = TILESYMBOLS[abs(tile_color + 1)]
        return symbol, lerp_color(scolor, GREY, greyshift)
    if tile_color <= 5:
        return TILESYMBOLS[tile_color]
    if tile_color >= 6:
        raise ValueError(
            "Cannot properly return tile colors greater than five!",
        )
    return None


def add_symbol_to_tile_surf(
    surf: pygame.surface.Surface,
    tilecolor: int,
    tilesize: int,
    greyshift: float = GREYSHIFT,
    font: Path = FONT,
) -> None:
    symbol, scolor = get_tile_symbol_and_color(tilecolor, greyshift)
    pyfont = pygame.font.Font(font, math.floor(math.sqrt(tilesize**2 * 2)) - 1)

    symbolsurf = pyfont.render(symbol, True, scolor)
    symbolsurf = auto_crop_clear(symbolsurf)
    ##    symbolsurf = pygame.transform.scale(symbolsurf, (tilesize, tilesize))

    ##    sw, sh = symbolsurf.get_size()
    ##
    ##    w, h = surf.get_size()
    ##
    ##    x = w/2 - sw/2
    ##    y = h/2 - sh/2
    ##    b = (round(x), round(y))

    sw, sh = symbolsurf.get_rect().center
    w, h = surf.get_rect().center
    x = w - sw
    y = h - sh

    surf.blit(symbolsurf, (int(x), int(y)))


##    surf.blit(symbolsurf, (0, 0))


@lru_cache
def get_tile_image(
    tile: Tile,
    tilesize: int,
    greyshift: float = GREYSHIFT,
    outlineSize: float = 0.2,
    font: Path = FONT,
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
            outlineSize,
        )
        # Add tile symbol
        add_symbol_to_tile_surf(surf, cid, tilesize, greyshift, font)

        return surf
    return make_square_surf(color, tilesize)
    # Add tile symbol
    ##    add_symbol_to_tile_surf(surf, cid, tilesize, greyshift, font)


def set_alpha(
    surface: pygame.surface.Surface,
    alpha: int,
) -> pygame.surface.Surface:
    """Return a surface by replacing the alpha channel of it with given alpha value, preserve color."""
    surface = surface.copy().convert_alpha()
    w, h = surface.get_size()
    for y in range(h):
        for x in range(w):
            r, g, b = surface.get_at((x, y))[:3]
            surface.set_at((x, y), pygame.Color(r, g, b, alpha))
    return surface


@lru_cache
def get_tile_container_image(
    wh: tuple[int, int],
    back: (
        pygame.color.Color
        | int
        | str
        | tuple[int, int, int]
        | tuple[int, int, int, int]
        | Sequence[int]
    ),
) -> pygame.surface.Surface:
    """Return a tile container image from a width and a height and a background color, and use a game's cache to help."""
    image = pygame.surface.Surface(wh)
    image.convert_alpha()
    image = set_alpha(image, 0)

    if back is not None:
        image.convert()
        image.fill(back)
    return image


class Font:
    """Font object, simplify using text."""

    def __init__(
        self,
        font_name: str,
        fontsize: int = 20,
        color: tuple[int, int, int] = (0, 0, 0),
        cx: bool = True,
        cy: bool = True,
        antialias: bool = False,
        background: tuple[int, int, int] | None = None,
        do_cache: bool = True,
    ) -> None:
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
        return "Font(%r, %i, %r, %r, %r, %r, %r, %r)" % (
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
        background: tuple[int, int, int] | str | None = None,
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

    ##    __slots__ = ("objects", "next_id", "cache")

    def __init__(self) -> None:
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

    def get_object(self, objectId: int) -> Object | None:
        """Return the object associated with object id given. Return None if object not found."""
        if objectId in self.objects:
            return self.objects[objectId]
        return None

    def get_objects_with_attr(self, attribute: str) -> tuple[Object, ...]:
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
    ) -> tuple[Object, ...]:
        """Return a tuple of object ids with <attribute> that are equal to <value>."""
        matches = []
        for oid in self.get_objects_with_attr(attribute):
            if getattr(self.objects[oid], attribute) == value:
                matches.append(oid)
        return tuple(matches)

    def get_object_given_name(self, name: str) -> tuple[Object, ...]:
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
        new: list[int] = [revnew[key] for key in sorted(revnew)]
        self._render_order = tuple(new)

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
        "name",
        "image",
        "location",
        "wh",
        "hidden",
        "location_mode_on_resize",
        "id",
        "screen_size_last",
        "game",
    )

    def __init__(self, name: str) -> None:
        """Sets self.name to name, and other values for rendering.

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

    def __repr__(self) -> str:
        """Return {self.name}()."""
        return f"{self.__class__.__name__}()"

    def getImageZero_noFix(self) -> tuple[float, float]:
        """Return the screen location of the topleft point of self.image."""
        return (
            self.location[0] - self.wh[0] / 2,
            self.location[1] - self.wh[1] / 2,
        )

    def get_image_zero(self) -> tuple[int, int]:
        """Return the screen location of the topleft point of self.image fixed to integer values."""
        x, y = self.getImageZero_noFix()
        return int(x), int(y)

    def get_rect(self) -> Rect:
        """Return a Rect object representing this Object's area."""
        return Rect(self.get_image_zero(), self.wh)

    def point_intersects(self, screen_location: tuple[int, int]) -> bool:
        """Return True if this Object intersects with a given screen location."""
        return self.get_rect().collidepoint(screen_location)

    def to_image_surface_location(
        self,
        screen_location: tuple[int, int],
    ) -> tuple[int, int]:
        """Return the location a screen location would be at on the objects image. Can return invalid data."""
        # Get zero zero in image locations
        zx, zy = self.get_image_zero()  # Zero x and y
        sx, sy = screen_location  # Screen x and y
        return sx - zx, sy - zy  # Location with respect to image dimensions

    def process(self, time_passed: float) -> None:
        """Process Object. Replace when calling this class."""

    def render(self, surface: pygame.surface.Surface) -> None:
        """Render self.image to surface if self.image is not None. Updates self.wh."""
        if self.image is None or self.hidden:
            return
        self.wh = self.image.get_size()
        x, y = self.get_image_zero()
        surface.blit(self.image, (int(x), int(y)))

    ##        pygame.draw.rect(surface, MAGENTA, self.get_rect(), 1)

    def __del__(self) -> None:
        """Delete self.image."""
        del self.image

    def screen_size_update(self) -> None:
        """Function called when screensize is changed."""
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

        self._lastloc: tuple[int, int] | None = None
        self._lasthidden: bool | None = None

    def reset_position(self) -> None:
        """Reset the position of all objects within."""
        raise NotImplementedError

    def get_intersection(
        self,
        point: tuple[int, int],
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
        Object.__del__(self)
        ObjectHandler.__del__(self)


class Tile(NamedTuple):
    """Represents a Tile."""

    color: int


class TileRenderer(Object):
    """Base class for all objects that need to render tiles."""

    greyshift = GREYSHIFT
    tile_size = TILESIZE

    def __init__(
        self,
        name: str,
        game: Any,
        tile_seperation: int | Literal["Auto"] = "Auto",
        background: tuple[int, int, int] = TILEDEFAULT,
    ) -> None:
        """Initialize renderer. Needs a game object for its cache and optional tile separation value and background RGB color.

        Defines the following attributes during initialization and uses throughout:
         self.game
         self.wh
         self.tileSep
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

        if tile_seperation == "Auto":
            self.tileSep = self.tile_size / 3.75
        else:
            self.tileSep = tile_seperation

        self.tile_full = self.tile_size + self.tileSep
        self.back = background

        self.image_update = True

    def get_rect(self) -> Rect:
        """Return a Rect object representing this row's area."""
        wh = self.wh[0] - self.tileSep * 2, self.wh[1] - self.tileSep * 2
        location = self.location[0] - wh[0] / 2, self.location[1] - wh[1] / 2
        return Rect(location, wh)

    def clear_image(self, tile_dimensions: tuple[int, int]) -> None:
        """Reset self.image using tile_dimensions tuple and fills with self.back. Also updates self.wh."""
        tw, th = tile_dimensions
        self.wh = Vector2(
            round(tw * self.tile_full + self.tileSep),
            round(th * self.tile_full + self.tileSep),
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
        self.image.blit(
            surf,
            (
                round(x * self.tile_full + self.tileSep),
                round(y * self.tile_full + self.tileSep),
            ),
        )

    def update_image(self) -> None:
        """Called when processing image changes, directed by self.image_update being True."""
        raise NotImplementedError

    def process(self, time_passed: float) -> None:
        """Call self.update_image() if self.image_update is True, then set self.update_image to False."""
        if self.image_update:
            self.update_image()
            self.image_update = False


class Cursor(TileRenderer):
    """Cursor Object."""

    greyshift = GREYSHIFT
    Render_Priority = "last"

    def __init__(self, game: Game) -> None:
        """Initialize cursor with a game it belongs to."""
        super().__init__("Cursor", game, "Auto", None)

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
        l = len(self.tiles)
        if self.holding_number_one and not count_number_one:
            return l - 1
        return l

    def is_holding(self, count_number_one: bool = False) -> bool:
        """Return True if the mouse is dragging something."""
        return self.get_held_count(count_number_one) > 0

    def get_held_info(self, includeNumberOne=False):
        """Returns color of tiles are and number of tiles held."""
        if not self.is_holding(includeNumberOne):
            return None, 0
        return self.tiles[0], self.get_held_count(includeNumberOne)

    def process(self, time_passed: float) -> None:
        """Process cursor."""
        x, y = pygame.mouse.get_pos()
        x = saturate(x, 0, SCREENSIZE[0])
        y = saturate(y, 0, SCREENSIZE[1])
        self.location = (x, y)
        if self.image_update:
            if len(self.tiles):
                self.update_image()
            else:
                self.image = None
            self.image_update = False

    def force_hold(self, tiles) -> None:
        """Pretty much it's drag but with no constraints."""
        for tile in tiles:
            if tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.image_update = True

    def drag(self, tiles):
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
        number: int | Literal["All"] = "All",
        allowOneTile: bool = False,
    ) -> None:
        """Return all of the tiles the Cursor is carrying."""
        if self.is_holding(allowOneTile):
            if number == "All":
                number = self.get_held_count(allowOneTile)
            else:
                number = saturate(number, 0, self.get_held_count(allowOneTile))

            tiles = []
            for tile in (self.tiles.popleft() for i in range(number)):
                if tile.color == NUMBERONETILE:
                    if not allowOneTile:
                        self.tiles.append(tile)
                        continue
                tiles.append(tile)
            self.image_update = True

            self.holding_number_one = NUMBERONETILE in {
                tile.color for tile in self.tiles
            }
            return tiles
        return []

    def drop_one_tile(self):
        """If holding the number one tile, drop it (returns it)."""
        if self.holding_number_one:
            notOne = self.drop("All", False)
            one = self.drop(1, True)
            self.drag(notOne)
            self.holding_number_one = False
            return one[0]
        return None

    def get_data(self):
        """Return all the data that makes this object special."""
        data = super().get_data()
        tiles = [t.get_data() for t in self.tiles]
        data["Ts"] = tiles
        return data

    def from_data(self, data):
        """Update this Cursor object from data."""
        super().from_data(data)
        self.tiles.clear()
        self.drag([Tile.from_data(t) for t in self.tiles])


def gsc_bound_index(bounds_failure_return: object = None):
    """Return a decorator for any grid or grid subclass that will keep index positions within bounds."""

    def gsc_bounds_keeper(function):
        """Grid or Grid Subclass Decorator that keeps index positions within bounds, as long as index is first argument after self arg."""

        @wraps(function)
        def keep_within_bounds(self, index: tuple[int, int], *args, **kwargs):
            """Wrapper function that makes sure a position tuple (x, y) is valid."""
            x, y = index
            if x < 0 or x >= self.size[0]:
                return bounds_failure_return
            if y < 0 or y >= self.size[1]:
                return bounds_failure_return
            return function(self, index, *args, **kwargs)

        return keep_within_bounds

    return gsc_bounds_keeper


class Grid(TileRenderer):
    """Grid object, used for boards and parts of other objects."""

    def __init__(
        self,
        size,
        game,
        tile_seperation="Auto",
        background=TILEDEFAULT,
    ):
        """Grid Objects require a size and game at least."""
        TileRenderer.__init__(self, "Grid", game, tile_seperation, background)

        self.size = tuple(size)

        self.data = array(
            [Tile(-6) for i in range(int(self.size[0] * self.size[1]))],
        ).reshape(self.size)

    def update_image(self):
        """Update self.image."""
        self.clear_image(self.size)

        for y in range(self.size[1]):
            for x in range(self.size[0]):
                self.render_tile(self.data[x, y], (x, y))

    def get_tile_point(self, screen_location):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.point_intersects(screen_location):
            return None
        # Otherwise, find out where screen point is in image locations
        bx, by = self.to_image_surface_location(
            screen_location,
        )  # board x and y
        # Finally, return the full divides (no decimals) of xy location by self.tile_full.
        return int(bx // self.tile_full), int(by // self.tile_full)

    @gsc_bound_index()
    def place_tile(self, xy: tuple[int, int], tile: Tile) -> bool:
        """Place a Tile Object if permitted to do so. Return True if success."""
        x, y = xy
        if self.data[x, y].color < 0:
            self.data[x, y] = tile
            del tile
            self.image_update = True
            return True
        return False

    @gsc_bound_index()
    def get_tile(self, xy: tuple[int, int], replace: int = -6) -> Tile | None:
        """Return a Tile Object from a given position in the grid if permitted. Return None on failure."""
        x, y = xy
        tile_copy = self.data[x, y]
        if tile_copy.color < 0:
            return None
        self.data[x, y] = Tile(replace)
        self.image_update = True
        return tile_copy

    @gsc_bound_index()
    def get_info(self, xy: tuple[int, int]) -> Tile:
        """Return the Tile Object at a given position without deleting it from the Grid."""
        x, y = xy
        return self.data[x, y]

    def get_colors(self):
        """Return a list of the colors of tiles within self."""
        colors = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                info_tile = self.get_info((x, y))
                if info_tile.color not in colors:
                    colors.append(info_tile.color)
        return colors

    def is_empty(self, empty_color=-6):
        """Return True if Grid is empty (all tiles are empty_color)."""
        colors = self.get_colors()
        # Colors should only be [-6] if empty
        return colors == [empty_color]

    def get_data(self):
        """Return data that makes this Grid Object special. Compress tile data by getting color values plus seven, then getting the hex of that as a string."""
        data = super().get_data()
        data["w"] = int(self.size[0])
        data["h"] = int(self.size[1])
        tiles = [
            f"{self.get_info((x, y)).color+7:x}"
            for x in range(self.size[0])
            for y in range(self.size[1])
        ]
        data["Ts"] = "".join(tiles)
        return data

    def from_data(self, data):
        """Update data in this board object."""
        super().from_data(data)
        self.size = int(data["w"]), int(data["h"])
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                c = data["Ts"][x + y]
                self.data[x, y] = Tile(int(f"0x{c}", 16) - 7)
        self.update_image = True

    def __del__(self):
        super().__del__()
        del self.data


class Board(Grid):
    """Represents the board in the Game."""

    size = (5, 5)
    bcolor = ORANGE

    def __init__(self, player, variant_play=False):
        """Requires a player object."""
        Grid.__init__(self, self.size, player.game, background=self.bcolor)
        self.name = "Board"
        self.player = player

        self.variant_play = variant_play
        self.additions = {}

        self.wall_tileing = False

    def __repr__(self):
        return f"Board({self.player!r}, {self.variant_play})"

    def setColors(self, keepReal=True):
        """Reset tile colors."""
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if not keepReal or self.data[x, y].color < 0:
                    self.data[x, y].color = -(
                        (self.size[1] - y + x) % REGTILECOUNT + 1
                    )

    ##                print(self.data[x, y].color, end=' ')
    ##            print()
    ##        print('-'*10)

    def get_row(self, index):
        """Return a row from self. Does not delete data from internal grid."""
        return [self.get_info((x, index)) for x in range(self.size[0])]

    def get_column(self, index):
        """Return a column from self. Does not delete data from internal grid."""
        return [self.get_info((index, y)) for y in range(self.size[1])]

    def get_colorsInRow(self, index, excludeNegs=True):
        """Return the colors placed in a given row in internal grid."""
        rowColors = [tile.color for tile in self.get_row(index)]
        if excludeNegs:
            rowColors = [c for c in rowColors if c >= 0]
        ccolors = Counter(rowColors)
        return sorted(ccolors.keys())

    def get_colorsInColumn(self, index, excludeNegs=True):
        """Return the colors placed in a given row in internal grid."""
        columnColors = [tile.color for tile in self.get_column(index)]
        if excludeNegs:
            columnColors = [c for c in columnColors if c >= 0]
        ccolors = Counter(columnColors)
        return sorted(ccolors.keys())

    def is_wall_tileing(self):
        """Return True if in Wall Tiling Mode."""
        return self.wall_tileing

    def get_tile_for_cursor_by_row(self, row):
        """Return A COPY OF tile the mouse should hold. Returns None on failure."""
        if row in self.additions:
            data = self.additions[row]
            if isinstance(data, Tile):
                return data
        return None

    @gsc_bound_index(False)
    def canPlaceTileColorAtPoint(self, position, tile):
        """Return True if tile's color is valid at given position."""
        column, row = position
        colors = set(
            self.get_colorsInColumn(column) + self.get_colorsInRow(row),
        )
        return tile.color not in colors

    def get_rowsToTile(self):
        """Return a dictionary of row numbers and row color to be wall tiled."""
        rows = {}
        for row in self.additions:
            if isinstance(self.additions[row], Tile):
                rows[row] = self.additions[row].color
        return rows

    def getValidSpacesForTileRow(self, row):
        """Return the valid drop columns of the additions tile for a given row."""
        valid = []
        if row in self.additions:
            tile = self.additions[row]
            if isinstance(tile, Tile):
                for column in range(self.size[0]):
                    if self.canPlaceTileColorAtPoint((column, row), tile):
                        valid.append(column)
                return tuple(valid)
        return ()

    def removeInvalidAdditions(self):
        """Remove invalid additions that would not be placeable."""
        # In the wall-tiling phase, it may happen that you
        # are not able to move the rightmost tile of a certain
        # pattern line over to the wall because there is no valid
        # space left for it. In this case, you must immediately
        # place all tiles of that pattern line in your floor line.
        for row in range(self.size[1]):
            if isinstance(self.additions[row], Tile):
                valid = self.getValidSpacesForTileRow(row)
                if not valid:
                    floor = self.player.get_object_by_name("FloorLine")
                    floor.place_tile(self.additions[row])
                    self.additions[row] = None

    @gsc_bound_index(False)
    def wall_tile_from_point(self, position):
        """Given a position, wall tile. Return success on placement. Also updates if in wall tiling mode."""
        success = False
        column, row = position
        at_point = self.get_info(position)
        if at_point.color <= 0 and row in self.additions:
            tile = self.additions[row]
            if tile is not None:
                if self.canPlaceTileColorAtPoint(position, tile):
                    self.place_tile(position, tile)
                    self.additions[row] = column
                    # Update invalid placements after new placement
                    self.removeInvalidAdditions()
                    success = True
        if not self.get_rowsToTile():
            self.wall_tileing = False
        return success

    def wall_tiling_mode(self, movedDict):
        """Set self into Wall Tiling Mode. Finishes automatically if not in variant play mode."""
        self.wall_tileing = True
        for key, value in ((key, movedDict[key]) for key in movedDict):
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
                    if tile is None:
                        continue
                    negTileColor = -(tile.color + 1)
                    if negTileColor in rowdata:
                        column = rowdata.index(negTileColor)
                        self.place_tile((column, row), tile)
                        # Set data to the column placed in, use for scoring
                        self.additions[row] = column
                    else:
                        raise RuntimeError(
                            "%i not in row %i!" % (negTileColor, row),
                        )
                else:
                    raise RuntimeError(f"{row} not in movedDict!")
            self.wall_tileing = False
        else:
            # Invalid additions can only happen in variant play mode.
            self.removeInvalidAdditions()

    @gsc_bound_index(([], []))
    def get_touches_continuous(self, xy):
        """Return two lists, each of which contain all the tiles that touch the tile at given x y position, including that position."""
        rs, cs = self.size
        x, y = xy
        # Get row and column tile color data
        row = [tile.color for tile in self.get_row(y)]
        column = [tile.color for tile in self.get_column(x)]

        # Both
        def gt(v, size, data):
            """Go through data forward and backward from point v out by size, and return all points from data with a value >= 0."""

            def trng(rng, data):
                """Try range. Return all of data in range up to when indexed value is < 0."""
                ret = []
                for tv in rng:
                    if data[tv] < 0:
                        break
                    ret.append(tv)
                return ret

            nt = trng(reversed(range(v)), data)
            pt = trng(range(v + 1, size), data)
            return nt + pt

        # Combine two lists by zipping together and returning list object.
        def comb(one, two):
            return list(zip(one, two, strict=False))

        # Return all of the self.get_info points for each value in lst.
        def get_all(lst):
            return [self.get_info(pos) for pos in lst]

        # Get row touches
        rowTouches = comb(gt(x, rs, row), [y] * rs)
        # Get column touches
        columnTouches = comb([x] * cs, gt(y, cs, column))
        # Get real tiles from indexes and return
        return get_all(rowTouches), get_all(columnTouches)

    def score_additions(self):
        """Using self.additions, which is set in self.wall_tiling_mode(), return the number of points the additions scored."""
        score = 0
        for x, y in ((self.additions[y], y) for y in range(self.size[1])):
            if x is not None:
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

    def get_filled_rows(self):
        """Return the number of filled rows on this board."""
        count = 0
        for row in range(self.size[1]):
            real = (t.color >= 0 for t in self.get_row(row))
            if all(real):
                count += 1
        return count

    def has_filled_row(self):
        """Return True if there is at least one completely filled horizontal line."""
        return self.get_filled_rows() >= 1

    def get_filled_columns(self):
        """Return the number of filled rows on this board."""
        count = 0
        for column in range(self.size[0]):
            real = (t.color >= 0 for t in self.get_column(column))
            if all(real):
                count += 1
        return count

    def get_filled_colors(self):
        """Return the number of completed colors on this board."""
        tiles = (
            self.get_info((x, y))
            for x in range(self.size[0])
            for y in range(self.size[1])
        )
        colors = [t.color for t in tiles]
        colorCount = Counter(colors)
        count = 0
        for fillNum in colorCount.values():
            if fillNum >= 5:
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
            self.setColors(True)
        super().process(time_passed)


class Row(TileRenderer):
    """Represents one of the five rows each player has."""

    greyshift = GREYSHIFT

    def __init__(self, player, size, tilesep="Auto", background=None):
        TileRenderer.__init__(self, "Row", player.game, tilesep, background)
        self.player = player
        self.size = int(size)

        self.color = -6
        self.tiles = deque([Tile(self.color)] * self.size)

    def __repr__(self):
        return "Row(%r, %i, ...)" % (self.game, self.size)

    @classmethod
    def from_list(cls, player, iterable):
        """Return a new Row Object from a given player and an iterable of tiles."""
        lst = deque(iterable)
        obj = cls(player, len(lst))
        obj.color = None
        obj.tiles = lst
        return obj

    def update_image(self) -> None:
        """Update self.image."""
        self.clear_image((self.size, 1))

        for x in range(len(self.tiles)):
            self.render_tile(self.tiles[x], (x, 0))

    def get_tile_point(self, screen_location: tuple[int, int]) -> int | None:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        xy = Grid.get_tile_point(self, screen_location)
        if xy is None:
            return None
        x, y = xy
        return self.size - 1 - x

    def get_placed(self) -> int:
        """Return the number of tiles in self that are not fake tiles, like grey ones."""
        return len([tile for tile in self.tiles if tile.color >= 0])

    def get_placeable(self):
        """Return the number of tiles permitted to be placed on self."""
        return self.size - self.get_placed()

    def is_full(self):
        """Return True if this row is full."""
        return self.get_placed() == self.size

    def get_info(self, location):
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
        colorCorrect = tile.color >= 0 and tile.color < 5
        numCorrect = self.get_placeable() > 0

        board = self.player.get_object_by_name("Board")
        colorNotPresent = tile.color not in board.get_colorsInRow(
            self.size - 1,
        )

        return placeable and colorCorrect and numCorrect and colorNotPresent

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

    def wall_tile(self, addToDict, empty_color: int = -6) -> None:
        """Move tiles around and into add dictionary for the wall tiling phase of the game. Removes tiles from self."""
        if "toBox" not in addToDict:
            addToDict["toBox"] = []
        if not self.is_full():
            addToDict[str(self.size)] = None
            return
        self.color = empty_color
        addToDict[str(self.size)] = self.get_tile()
        for _i in range(self.size - 1):
            addToDict["toBox"].append(self.get_tile())

    def set_background(self, color) -> None:
        """Set the background color for this row."""
        self.back = color
        self.image_update = True


class PatternLine(MultipartObject):
    """Represents multiple rows to make the pattern line."""

    size = (5, 5)

    def __init__(self, player, row_seperation: int = 0) -> None:
        MultipartObject.__init__(self, "PatternLine")
        self.player = player
        self.rowSep = row_seperation

        for x, _y in zip(
            range(self.size[0]),
            range(self.size[1]),
            strict=True,
        ):
            self.add_object(Row(self.player, x + 1))

        self.set_background(None)

        self._lastloc = 0, 0

    def set_background(self, color):
        """Set the background color for all rows in the pattern line."""
        self.set_attr_all("back", color)
        self.set_attr_all("image_update", True)

    def get_row(self, row):
        """Return given row."""
        return self.get_object(row)

    def reset_position(self) -> None:
        """Reset Locations of Rows according to self.location."""
        last = self.size[1]
        w = self.get_row(last - 1).wh[0]
        if w is None:
            raise RuntimeError(
                "Image Dimensions for Row Object (row.wh) are None!",
            )
        h1 = self.get_row(0).tile_full
        h = last * h1
        self.wh = w, h
        w1 = h1 / 2

        x, y = self.location
        y -= h / 2 - w1
        for rid in self.objects:
            l = last - self.objects[rid].size
            self.objects[rid].location = x + (l * w1), y + rid * h1

    def get_tile_point(
        self,
        screen_location: tuple[int, int],
    ) -> tuple[int, int]:
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        for y in range(self.size[1]):
            x = self.get_row(y).get_tile_point(screen_location)
            if x is not None:
                return x, y
        return None

    def is_full(self) -> bool:
        """Return True if self is full."""
        return all(self.get_row(rid).is_full() for rid in range(self.size[1]))

    def wall_tileing(self):
        """Return a dictionary to be used with wall tiling. Removes tiles from rows."""
        values = {}
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

    __slots__ = ("font", "_cxy", "_last")

    def __init__(
        self,
        font_size: int,
        color: tuple[int, int, int],
        background: tuple[int, int, int] | None = None,
        cx: bool = True,
        cy: bool = True,
        name: str = "",
    ) -> None:
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
        self._last = None

    def get_image_zero(self) -> tuple[int, int]:
        """Return the screen location of the topleft point of self.image."""
        x = self.location[0]
        y = self.location[1]
        if self._cxy[0]:
            x -= self.wh[0] // 2
        if self._cxy[1]:
            y -= self.wh[1] // 2
        return x, y

    def __repr__(self) -> str:
        return "<Text Object>"

    @staticmethod
    def get_font_height(font: str, size: int) -> int:
        """Return the height of font at fontsize size."""
        return pygame.font.Font(font, size).get_height()

    def update_value(
        self,
        text: str | None,
        size: int | None = None,
        color: tuple[int, int, int] | None = None,
        background: tuple[int, int, int] | str = "set",
    ) -> pygame.surface.Surface:
        """Return a surface of given text rendered in FONT."""
        if background == "set":
            self.image = self.font.render_nosurf(text, size, color)
            return self.image
        self.image = self.font.render_nosurf(text, size, color, background)
        return self.image

    def get_surface(self) -> pygame.surface.Surface:
        """Return self.image."""
        return self.image

    def get_tile_point(self, location):
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

    def __init__(self, player) -> None:
        Row.__init__(self, player, self.size, background=ORANGE)
        self.name = "FloorLine"

        ##        self.font = Font(FONT, round(self.tile_size*1.2), color=BLACK, cx=False, cy=False)
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
        return f"FloorLine({self.player!r})"

    def render(self, surface: pygame.surface.Surface) -> None:
        """Update self.image."""
        Row.render(self, surface)

        sx, sy = self.location
        if self.wh is None:
            return
        w, h = self.wh
        for x in range(self.size):
            xy = round(x * self.tile_full + self.tileSep + sx - w / 2), round(
                self.tileSep + sy - h / 2,
            )
            self.text.update_value(str(self.numbers[x]))
            self.text.location = xy
            self.text.render(surface)

    ##            self.font.render(surface, str(self.numbers[x]), xy)

    def place_tile(self, tile: Tile) -> None:
        """Place a given Tile Object on self if permitted."""
        self.tiles.insert(self.get_placed(), tile)

        if tile.color == self.number_one_color:
            self.has_number_one_tile = True

        boxLid = self.player.game.get_object_by_name("BoxLid")

        def handleEnd(end: Tile) -> None:
            """Handle the end tile we are replacing. Ensures number one tile is not removed."""
            if not end.color < 0:
                if end.color == self.number_one_color:
                    handleEnd(self.tiles.pop())
                    self.tiles.appendleft(end)
                    return
                boxLid.add_tile(end)

        handleEnd(self.tiles.pop())

        self.image_update = True

    def score_tiles(self) -> int:
        """Score self.tiles and return how to change points."""
        running_total = 0
        for x in range(self.size):
            if self.tiles[x].color >= 0:
                running_total += self.numbers[x]
            elif x < self.size - 1:
                if self.tiles[x + 1].color >= 0:
                    raise RuntimeError(
                        "Player is likely cheating! Invalid placement of FloorLine tiles!",
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
    outSize = 0.1

    def __init__(self, game: Game, factory_id: int) -> None:
        """Initialize factory."""
        super().__init__(self.size, game, background=None)
        self.number = factory_id
        self.name = f"Factory{self.number}"

        self.radius = math.ceil(
            self.tile_full * self.size[0] * self.size[1] / 3 + 3,
        )

    def __repr__(self) -> str:
        return "Factory(%r, %i)" % (self.game, self.number)

    def add_circle(self, surface: pygame.surface.Surface) -> None:
        if f"FactoryCircle{self.radius}" not in self.game.cache:
            rad = math.ceil(self.radius)
            surf = set_alpha(pygame.surface.Surface((2 * rad, 2 * rad)), 1)
            pygame.draw.circle(surf, self.outline, (rad, rad), rad)
            pygame.draw.circle(
                surf,
                self.color,
                (rad, rad),
                math.ceil(rad * (1 - self.outSize)),
            )
            self.game.cache[f"FactoryCircle{self.radius}"] = surf
        surf = self.game.cache[f"FactoryCircle{self.radius}"].copy()
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
            raise RuntimeError(
                "Insufficiant quantity of tiles! Needs %i!"
                % self.size[0]
                * self.size[1],
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
                self.get_tile((x, y))
                for x in range(self.size[0])
                for y in range(self.size[1])
            )
            if tile.color != -6
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
            self.radius = self.tile_full * self.size[0] * self.size[1] / 3 + 3
        super().process(time_passed)


class Factories(MultipartObject):
    """Factories Multipart Object, made of multiple Factory Objects."""

    teach = 4

    def __init__(
        self,
        game: Game,
        factories: int,
        size: int | Literal["Auto"] = "Auto",
    ) -> None:
        """Requires a number of factories."""
        super().__init__("Factories")

        self.game = game
        self.count = factories

        for i in range(self.count):
            self.add_object(Factory(self.game, i))

        if size == "Auto":
            self.objects[0].process(0)
            rad = self.objects[0].radius
            self.size = rad * 5
        else:
            self.size = size
        self.size = math.ceil(self.size)

        self.play_tiles_from_bag()

    def __repr__(self) -> str:
        return "Factories(%r, %i, ...)" % (self.game, self.count)

    def reset_position(self) -> None:
        """Reset the position of all factories within."""
        degrees = 360 / self.count
        for i in range(self.count):
            rot = math.radians(degrees * i)
            self.objects[i].location = (
                math.sin(rot) * self.size + self.location[0],
                math.cos(rot) * self.size + self.location[1],
            )

    def process(self, time_passed: float) -> None:
        """Process factories. Does not react to cursor if hidden."""
        super().process(time_passed)
        if not self.hidden:
            cursor = self.game.get_object_by_name("Cursor")
            assert isinstance(cursor, Cursor)
            if cursor.is_pressed() and not cursor.is_holding():
                obj, point = self.get_intersection(cursor.location)
                if obj is not None and point is not None:
                    oid = int(obj[7:])
                    tileAtPoint = self.objects[oid].get_info(point)
                    if (tileAtPoint is not None) and tileAtPoint.color >= 0:
                        table = self.game.get_object_by_name("TableCenter")
                        assert isinstance(table, Table)
                        select, tocenter = self.objects[oid].grab_color(
                            tileAtPoint.color,
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
                    drawn.append(self.game.bag.draw_tile())
                else:  # Otherwise, get the box lid
                    boxLid = self.game.get_object_by_name("BoxLid")
                    # If the box lid is not empty,
                    if not boxLid.is_empty():
                        # Add all the tiles from the box lid to the bag
                        self.game.bag.add_tiles(boxLid.get_tiles())
                        # and shake the bag to randomize everything
                        self.game.bag.reset()
                        # Then, grab a tile from the bag like usual.
                        drawn.append(self.game.bag.draw_tile())
                    else:
                        # "In the rare case that you run out of tiles again
                        # while there are none left in the lid, start a new
                        # round as usual even though are not all factory
                        # displays are properly filled."
                        drawn.append(Tile(empty_color))
            # Place drawn tiles on factory
            self.objects[fid].fill(drawn)

    def is_all_empty(self) -> bool:
        """Return True if all factories are empty."""
        return all(self.objects[fid].is_empty() for fid in range(self.count))


class TableCenter(Grid):
    """Object that represents the center of the table."""

    size = (6, 6)
    first_tile_color = NUMBERONETILE

    def __init__(self, game: Game, has_number_one_tile: bool = True) -> None:
        """Requires a game object handler to exist in."""
        Grid.__init__(self, self.size, game, background=None)
        self.game = game
        self.name = "TableCenter"

        self.number_one_tile_exists = False
        if has_number_one_tile:
            self.add_number_one_tile()

        self.next_position = (0, 0)

    def __repr__(self) -> None:
        return f"TableCenter({self.game!r})"

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
                    if self.get_info((x, y)).color == self.first_tile_color:
                        continue
                at = self.get_tile((x, y), replace)

                if at is not None:
                    full.append(at)
        sortedTiles = sorted(full, key=sort_tiles)
        self.next_position = (0, 0)
        self.add_tiles(sortedTiles, False)

    def pull_tiles(self, tile_color: int, replace: int = -6) -> list[Tile]:
        """Remove all of the tiles of tile_color from the Table Center Grid."""
        to_pull: list[tuple[int, int]] = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                info_tile = self.get_info((x, y))
                if info_tile.color == tile_color:
                    to_pull.append((x, y))
                elif self.number_one_tile_exists:
                    if info_tile.color == self.first_tile_color:
                        to_pull.append((x, y))
                        self.number_one_tile_exists = False
        tiles = [self.get_tile(pos, replace) for pos in to_pull]
        self.reorder_tiles(replace)
        return tiles

    def process(self, time_passed: float) -> None:
        """Process factories."""
        if not self.hidden:
            cursor = self.game.get_object_by_name("Cursor")
            if (
                cursor.is_pressed()
                and not cursor.is_holding()
                and not self.is_empty()
            ):
                if self.point_intersects(cursor.location):
                    point = self.get_tile_point(cursor.location)
                    # Shouldn't return none anymore since we have point_intersects now.
                    colorAtPoint = self.get_info(point).color
                    if colorAtPoint >= 0 and colorAtPoint < 5:
                        cursor.drag(self.pull_tiles(colorAtPoint))
        super().process(time_passed)


class Bag:
    """Represents the bag full of tiles."""

    __slots__ = (
        "tile_count",
        "tile_types",
        "tile_names",
        "percent_each",
        "tiles",
    )

    def __init__(self, tile_count: int = 100, tile_types: int = 5) -> None:
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
                **{
                    tile_name: self.percent_each
                    for tile_name in self.tile_names
                },
            ),
        )

    def __repr__(self) -> str:
        return "Bag(%i, %i)" % (self.tile_count, self.tile_types)

    def reset(self) -> None:
        """Randomize all the tiles in the bag."""
        self.tiles = deque(randomize(self.tiles))

    def get_color(self, tile_name):
        """Return the color of a named tile."""
        if tile_name not in self.tile_names:
            raise ValueError(f"Tile Name {tile_name} Not Found!")
        return self.tile_names.index(tile_name)

    def get_tile(self, tile_name) -> Tile:
        """Return a Tile Object from a tile name."""
        return Tile(self.get_color(tile_name))

    def get_count(self) -> int:
        """Return number of tiles currently held."""
        return len(self.tiles)

    def is_empty(self) -> bool:
        """Return True if no tiles are currently held."""
        return self.get_count() == 0

    def draw_tile(self):
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
        rnge = (0, len(self.tiles) - 1)
        if rnge[1] - rnge[0] <= 1:
            index = 0
        else:
            index = random.randint(rnge[0], rnge[1])
        ##        self.tiles.insert(random.randint(0, len(self.tiles)-1), self.get_name(int(tile_object.color)))
        self.tiles.insert(index, name)
        del tile_object

    def add_tiles(self, tile_objects: Iterable[Tile]) -> None:
        """Add multiple Tile Objects to the bag."""
        for tile_object in tile_objects:
            self.add_tile(tile_object)


class BoxLid(Object):
    """BoxLid Object, represents the box lid were tiles go before being added to the bag again."""

    def __init__(self, game: Game) -> None:
        Object.__init__(self, "BoxLid")
        self.game = game
        self.tiles = deque()

    def __repr__(self) -> str:
        return f"BoxLid({self.game!r})"

    def add_tile(self, tile: Tile) -> None:
        """Add a tile to self."""
        if tile.color >= 0 and tile.color < 5:
            self.tiles.append(tile)
        else:
            raise Warning(
                f"BoxLid.add_tile tried to add an invalid tile to self ({tile.color}). Be careful, bad things might be trying to happen.",
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
    ):
        """Requires a player Id and can be told to be controlled by the network or be in variant play mode."""
        MultipartObject.__init__(self, "Player%i" % player_id)

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
        self.is_wall_tileing = False
        self.just_held = False
        self.just_dropped = False

        self.update_score()

        self._lastloc = 0, 0

    def __repr__(self) -> str:
        return "Player(%r, %i, %s, %s)" % (
            self.game,
            self.player_id,
            self.networked,
            self.varient_play,
        )

    def update_score(self) -> None:
        """Update the scorebox for this player."""
        score_box = self.get_object_by_name("Text")
        score_box.update_value(f"Player {self.player_id+1}: {self.score}")

    def trigger_turn_now(self) -> None:
        """It is this player's turn now."""
        if not self.is_turn:
            pattern_line = self.get_object_by_name("PatternLine")
            if self.is_wall_tileing:
                board = self.get_object_by_name("Board")
                rows = board.get_rowsToTile()
                for rowpos in rows:
                    pattern_line.get_row(rowpos).set_background(
                        get_tile_color(rows[rowpos], board.greyshift),
                    )
            else:
                pattern_line.set_background(PATSELECTCOLOR)
        self.is_turn = True

    def end_of_turn(self) -> None:
        """It is no longer this player's turn."""
        if self.is_turn:
            pattern_line = self.get_object_by_name("PatternLine")
            pattern_line.set_background(None)
        self.is_turn = False

    def end_of_game_trigger(self) -> None:
        """Function called by end state when game is over; Hide pattern lines and floor line."""
        pattern = self.get_object_by_name("PatternLine")
        floor = self.get_object_by_name("FloorLine")

        pattern.hidden = True
        floor.hidden = True

    def reset_position(self) -> bool:
        """Reset positions of all parts of self based off self.location."""
        x, y = self.location

        bw, bh = self.get_object_by_name("Board").wh
        self.get_object_by_name("Board").location = x + bw / 2, y
        lw = self.get_object_by_name("PatternLine").wh[0] / 2
        self.get_object_by_name("PatternLine").location = x - lw, y
        self.get_object_by_name("FloorLine").wh[0]
        self.get_object_by_name("FloorLine").location = x - lw * (
            2 / 3
        ) + TILESIZE / 3.75, y + bh * (2 / 3)
        self.get_object_by_name("Text").location = x - (bw / 3), y - (
            bh * (2 / 3)
        )

    def wall_tileing(self) -> None:
        """Do the wall tiling phase of the game for this player."""
        self.is_wall_tileing = True
        pattern_line = self.get_object_by_name("PatternLine")
        self.get_object_by_name("FloorLine")
        board = self.get_object_by_name("Board")
        boxLid = self.game.get_object_by_name("BoxLid")

        data = pattern_line.wall_tileing()
        boxLid.add_tiles(data["toBox"])
        del data["toBox"]

        board.wall_tiling_mode(data)

    def done_wall_tileing(self) -> bool:
        """Return True if internal Board is done wall tiling."""
        board = self.get_object_by_name("Board")
        return not board.is_wall_tileing()

    def next_round(self) -> None:
        """Called when player is done wall tiling."""
        self.is_wall_tileing = False

    def score_phase(self):
        """Do the scoring phase of the game for this player."""
        board = self.get_object_by_name("Board")
        floorLine = self.get_object_by_name("FloorLine")
        boxLid = self.game.get_object_by_name("BoxLid")

        def saturatescore():
            if self.score < 0:
                self.score = 0

        self.score += board.score_additions()
        self.score += floorLine.score_tiles()
        saturatescore()

        toBox, number_one = floorLine.get_tiles()
        boxLid.add_tiles(toBox)

        self.update_score()

        return number_one

    def end_of_game_scoring(self) -> None:
        """Update final score with additional end of game points."""
        board = self.get_object_by_name("Board")

        self.score += board.end_of_game_scoreing()

        self.update_score()

    def has_horzontal_line(self) -> bool:
        """Return True if this player has a horizontal line on their game board filled."""
        board = self.get_object_by_name("Board")
        return board.has_filled_row()

    def get_horizontal_lines(self) -> int:
        """Return the number of filled horizontal lines this player has on their game board."""
        board = self.get_object_by_name("Board")
        return board.get_filled_rows()

    def process(self, time_passed: float) -> None:
        """Process Player."""
        if self.is_turn:  # Is our turn?
            if self.hidden and self.is_wall_tileing and self.varient_play:
                # If hidden, not anymore. Our turn.
                self.hidden = False
            if not self.networked:  # We not networked.
                cursor = self.game.get_object_by_name("Cursor")
                boxLid = self.game.get_object_by_name("BoxLid")
                pattern_line = self.get_object_by_name("PatternLine")
                floorLine = self.get_object_by_name("FloorLine")
                board = self.get_object_by_name("Board")
                if cursor.is_pressed():  # Mouse down?
                    obj, point = self.get_intersection(cursor.location)
                    if (
                        obj is not None and point is not None
                    ):  # Something pressed
                        if cursor.is_holding():  # Cursor holding tiles
                            move_made = False
                            if not self.is_wall_tileing:  # Is wall tiling:
                                if obj == "PatternLine":
                                    pos, row_number = point
                                    row = pattern_line.get_row(row_number)
                                    if not row.is_full():
                                        info = row.get_info(pos)
                                        if info is not None and info.color < 0:
                                            color, held = (
                                                cursor.get_held_info()
                                            )
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
                                elif obj == "FloorLine":
                                    tiles_to_add = cursor.drop()
                                    if floorLine.is_full():  # Floor is full,
                                        # Add tiles to box instead.
                                        boxLid.add_tiles(tiles_to_add)
                                    elif floorLine.get_placeable() < len(
                                        tiles_to_add,
                                    ):
                                        # Add tiles to floor line and then to box
                                        while len(tiles_to_add) > 0:
                                            if floorLine.get_placeable() > 0:
                                                floorLine.place_tile(
                                                    tiles_to_add.pop(),
                                                )
                                            else:
                                                boxLid.add_tile(
                                                    tiles_to_add.pop(),
                                                )
                                    else:  # Otherwise add to floor line for all.
                                        floorLine.place_tiles(tiles_to_add)
                                    move_made = True
                            elif (
                                not self.just_held
                            ):  # Cursor holding and wall tiling
                                if obj == "Board":
                                    at_point = board.get_info(point)
                                    if at_point.color == -6:
                                        column, row = point
                                        cursor_tile = cursor.drop(1)[0]
                                        board_tile = (
                                            board.get_tile_for_cursor_by_row(
                                                row,
                                            )
                                        )
                                        if board_tile is not None:
                                            if (
                                                cursor_tile.color
                                                == board_tile.color
                                            ):
                                                if board.wall_tile_from_point(
                                                    point,
                                                ):
                                                    self.just_dropped = True
                                                    pattern_line.get_row(
                                                        row,
                                                    ).set_background(None)

                            if move_made:
                                if not self.is_wall_tileing:
                                    if cursor.holding_number_one:
                                        floorLine.place_tile(
                                            cursor.drop_one_tile(),
                                        )
                                    if cursor.get_held_count(True) == 0:
                                        self.game.next_turn()
                        else:  # Mouse down, something pressed, and not holding anything
                            if (
                                self.is_wall_tileing
                            ):  # Wall tiling, pressed, not holding
                                if obj == "Board":
                                    if not self.just_dropped:
                                        column_number, row_number = point
                                        tile = (
                                            board.get_tile_for_cursor_by_row(
                                                row_number,
                                            )
                                        )
                                        if tile is not None:
                                            cursor.drag([tile])
                                            self.just_held = True
                else:  # Mouse up
                    if self.just_held:
                        self.just_held = False
                    if self.just_dropped:
                        self.just_dropped = False
            if self.is_wall_tileing and self.done_wall_tileing():
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
        super().__init__(font_size, self.textcolor, background=None)
        self.name = f"Button{name}"
        self.state = state

        self.minsize = int(minimum_size)
        self.update_value(initial_value)

        self.borderWidth = math.floor(font_size / 12)  # 5

        self.action = lambda: None
        self.delay = 0.6
        self.cur_time = 1

    def __repr__(self) -> str:
        return f"Button({self.name[6:]}, {self.state}, {self.minsize}, {self.font.last_text}, {self.font.pyfont})"

    def get_height(self) -> int:
        return self.font.get_height()

    def bind_action(self, function: Callable[[], None]) -> None:
        """When self is pressed, call given function exactly once. Function takes no arguments."""
        self.action = function

    def update_value(
        self,
        text: str,
        size: int | None = None,
        color: tuple[int, int, int] | None = None,
        background: str = "set",
    ) -> None:
        disp = str(text).center(self.minsize)
        super().update_value(f" {disp} ", size, color, background)
        self.font.last_text = disp

    def render(self, surface: pygame.surface.Surface) -> None:
        if not self.hidden:
            text_rect = self.get_rect()
            ##            if PYGAME_VERSION < 201:
            ##                pygame.draw.rect(surface, self.backcolor, text_rect)
            ##                pygame.draw.rect(surface, BLACK, text_rect, self.borderWidth)
            ##            else:
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
        cursor = self.state.game.get_object_by_name("Cursor")
        if not self.hidden and cursor.is_pressed():
            if self.point_intersects(cursor.location):
                return True
        return False

    def process(self, time_passed: float) -> None:
        """Call self.action one time when pressed, then wait self.delay to call again."""
        if self.cur_time > 0:
            self.cur_time = max(self.cur_time - time_passed, 0)
        else:
            if self.is_pressed():
                self.action()
                self.cur_time = self.delay
        if self.font.last_text != self._last:
            self.textSize = self.font.pyfont.size(f" {self.font.last_text} ")
        super().process(time_passed)


class GameState:
    """Base class for all game states."""

    name = "Base Class"

    def __init__(self, name):
        """Initialize state with a name, set self.game to None to be overwritten later."""
        self.game = None
        self.name = name

    def __repr__(self):
        return f"<GameState {self.name}>"

    def entry_actions(self):
        """Perform entry actions for this GameState."""

    def do_actions(self):
        """Perform actions for this GameState."""

    def check_state(self):
        """Check state and return new state. None remains in current state."""
        return

    def exit_actions(self):
        """Perform exit actions for this GameState."""


class MenuState(GameState):
    """Game State where there is a menu with buttons."""

    button_minimum = 10
    fontsize = BUTTONFONTSIZE

    def __init__(self, name: str) -> None:
        """Initialize GameState and set up self.bh."""
        super().__init__(name)
        self.bh = Text.get_font_height(FONT, self.fontsize)

        self.toState = None

    def add_button(
        self,
        name: str,
        value: str,
        action,
        location: Literal["Center"] | tuple[int, int] = "Center",
        size: int = fontsize,
        minlen: int = button_minimum,
    ) -> int:
        """Add a new Button object to self.game with arguments. Return button id."""
        button = Button(self, name, minlen, value, size)
        button.bind_action(action)
        if location != "Center":
            button.location = location
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
        text.location = location
        text.update_value(value)
        self.game.add_object(text)
        return text.id

    def entry_actions(self) -> None:
        """Clear all objects, add cursor object, and set up toState."""
        self.toState = None

        self.game.rm_star()
        self.game.add_object(Cursor(self.game))

    def set_var(self, attribute: str, value: object) -> None:
        """Set MenuState.{attribute} to {value}."""
        setattr(self, attribute, value)

    def to_state(self, state_name: str) -> Callable[[], None]:
        """Return a function that will change game state to state_name."""

        def to_state_name() -> None:
            """Set MenuState.toState to {state_name}."""
            self.toState = state_name

        return to_state_name

    def var_dependant_to_state(
        self,
        **kwargs: tuple[str, object],
    ) -> Callable[[], None]:
        """Attribute name = (target value, on trigger tostate)."""
        for state in kwargs:
            if not len(kwargs[state]) == 2:
                raise ValueError(f'Key "{state}" is invalid!')
            key, value = kwargs[state]
            if not hasattr(self, key):
                raise ValueError(
                    f'{self} object does not have attribute "{key}"!',
                )

        def to_state_by_attributes() -> None:
            """Set MenuState.toState to a new state if conditions are right."""
            for state in kwargs:
                key, value = kwargs[state]
                if getattr(self, key) == value:
                    self.toState = state

        return to_state_by_attributes

    def with_update(
        self,
        update_function: Callable[[], None],
    ) -> Callable[[Callable[[], None]], Callable[[], None]]:
        """Return a wrapper for a function that will call update_function after function."""

        def update_wrapper(function: Callable[[], None]) -> Callable[[], None]:
            """Wrapper for any function that could require a screen update."""

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
            text = self.game.get_object_by_name(f"Text{text_name}")
            text.update_value(value_function())

        return updater

    def toggle_button_state(
        self,
        textname: str,
        boolattr: str,
        textfunc,
    ) -> Callable[[], None]:
        """Return function that will toggle the value of text object <textname>, toggling attribute <boolattr>, and setting text value with textfunc."""

        def valfunc():
            """Return the new value for the text object. Gets called AFTER value is toggled."""
            return textfunc(getattr(self, boolattr))

        @self.with_update(self.update_text(textname, valfunc))
        def toggle_value() -> None:
            """Toggle the value of boolattr."""
            self.set_var(boolattr, not getattr(self, boolattr))

        return toggle_value

    def check_state(self) -> str | None:
        """Return self.toState."""
        return self.toState


class InitState(GameState):
    """Initialize state."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize self."""
        super().__init__("Init")

    def entry_actions(self) -> None:
        """Register keyboard handlers."""
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
            (sw / 2, sh / 2 - self.bh * 0.5),
        )
        self.add_button(
            "ToCredits",
            "Credits",
            self.to_state("Credits"),
            (sw / 2, sh / 2 + self.bh * 3),
            self.fontsize / 1.5,
        )
        self.add_button(
            "Quit",
            "Quit",
            self.game.raise_close,
            (sw / 2, sh / 2 + self.bh * 4),
            self.fontsize / 1.5,
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
            cx: bool,
            cy: bool,
        ) -> None:
            count = end - start + 1
            evencount = count % 2 == 0
            mid = count // 2

            def add_number(
                number: int,
                display: str | int,
                cx: bool,
                cy: bool,
            ) -> None:
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
                    (cx + (width_each * x), cy),
                    size=self.fontsize / 1.5,
                    minlen=3,
                )

            for i in range(count):
                add_number(i, start + i, cx, cy)

        sw, sh = SCREENSIZE
        cx = sw / 2
        cy = sh / 2

        def host_text(x):
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
            size=self.fontsize / 1.5,
        )

        # TEMPORARY: Hide everything to do with "Host Mode", networked games aren't done yet.
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
            size=self.fontsize / 1.5,
        )

        self.add_text(
            "Players",
            f"Players: {self.player_count}",
            (cx, cy + self.bh),
        )
        add_numbers(2, 4, 70, cx, cy + self.bh * 2)

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
        self.game.start_game(
            self.player_count,
            self.variant_play,
            self.host_mode,
        )
        self.game.bag.full_reset()


class PhaseFactoryOffer(GameState):
    """Game state when it's the Factory Offer Stage."""

    def __init__(self):
        super().__init__("FactoryOffer")

    def entry_actions(self):
        """Advance turn."""
        self.game.next_turn()

    def check_state(self):
        """If all tiles are gone, go to wall tiling. Otherwise keep waiting for that to happen."""
        fact = self.game.get_object_by_name("Factories")
        table = self.game.get_object_by_name("TableCenter")
        cursor = self.game.get_object_by_name("Cursor")
        if (
            fact.is_all_empty()
            and table.is_empty()
            and not cursor.is_holding(True)
        ):
            return "WallTiling"
        return None


class PhaseFactoryOfferNetworked(PhaseFactoryOffer):
    def __init__(self):
        GameState.__init__(self, "FactoryOfferNetworked")

    def check_state(self):
        return "WallTilingNetworked"


class PhaseWallTiling(GameState):
    def __init__(self):
        super().__init__("WallTiling")

    def entry_actions(self):
        self.next_starter = None
        self.not_processed = []

        self.game.player_turn_over()

        # For each player,
        for player_id in range(self.game.players):
            # Activate wall tiling mode.
            player = self.game.get_player(player_id)
            player.wall_tileing()
            # Add that player's player_id to the list of not-processed players.
            self.not_processed.append(player.player_id)

        # Start processing players.
        self.game.next_turn()

    def do_actions(self):
        if self.not_processed:
            if self.game.player_turn in self.not_processed:
                player = self.game.get_player(self.game.player_turn)
                if player.done_wall_tileing():
                    # Once player is done wall tiling, score their moves.
                    number_one = (
                        player.score_phase()
                    )  # Also gets if they had the number one tile.
                    if number_one:
                        # If player had the number one tile, remember that.
                        self.next_starter = self.game.player_turn
                        # Then, add the number one tile back to the table center.
                        table = self.game.get_object_by_name("TableCenter")
                        table.add_number_one_tile()
                    # After calculating their score, delete player from un-processed list
                    self.not_processed.remove(self.game.player_turn)
                    # and continue to the next un-processed player.
                    self.game.next_turn()
            else:
                self.game.next_turn()

    def check_state(self):
        cursor = self.game.get_object_by_name("Cursor")
        if not self.not_processed and not cursor.is_holding():
            return "PrepareNext"
        return None

    def exit_actions(self):
        # Set up the player that had the number one tile to be the starting player next round.
        self.game.player_turn_over()
        # Goal: make (self.player_turn + 1) % self.players = self.next_starter
        nturn = self.next_starter - 1
        if nturn < 0:
            nturn += self.game.players
        self.game.player_turn = nturn


class PhaseWallTilingNetworked(PhaseWallTiling):
    def __init__(self):
        GameState.__init__(self, "WallTilingNetworked")

    def check_state(self):
        return "PrepareNextNetworked"


class PhasePrepareNext(GameState):
    def __init__(self):
        super().__init__("PrepareNext")

    def entry_actions(self):
        players = (
            self.game.get_player(player_id)
            for player_id in range(self.game.players)
        )
        complete = (player.has_horzontal_line() for player in players)
        self.newRound = not any(complete)

    def do_actions(self):
        if self.newRound:
            fact = self.game.get_object_by_name("Factories")
            # This also handles bag re-filling from box lid.
            fact.play_tiles_from_bag()

    def check_state(self):
        if self.newRound:
            return "FactoryOffer"
        return "End"


class PhasePrepareNextNetworked(PhasePrepareNext):
    def __init__(self):
        GameState.__init__(self, "PrepareNextNetworked")

    def check_state(self):
        return "EndNetworked"


class EndScreen(MenuState):
    def __init__(self):
        super().__init__("End")
        self.ranking = {}
        self.wininf = ""

    def get_winners(self):
        """Update self.ranking by player scores."""
        self.ranking = {}
        scpid = {}
        for player_id in range(self.game.players):
            player = self.game.get_player(player_id)
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
                    key=lambda x: x[0],
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
            players = self.ranking[rank]
            cnt = len(players)
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
            text += line.format(*players)
        self.wininf = text[:-1]

    def entry_actions(self):
        # Figure out who won the game by points.
        self.get_winners()
        # Hide everything
        table = self.game.get_object_by_name("TableCenter")
        table.hidden = True

        fact = self.game.get_object_by_name("Factories")
        fact.set_attr_all("hidden", True)

        # Add buttons
        bid = self.add_button(
            "ReturnTitle",
            "Return to Title",
            self.to_state("Title"),
            (SCREENSIZE[0] / 2, math.floor(SCREENSIZE[1] * (4 / 5))),
        )
        buttontitle = self.game.get_object(bid)
        buttontitle.Render_Priority = "last-1"
        buttontitle.cur_time = 2

        # Add score board
        x = SCREENSIZE[0] / 2
        y = 10
        for idx, line in enumerate(self.wininf.split("\n")):
            self.add_text(f"Line{idx}", line, (x, y), cx=True, cy=False)
            ##            self.game.get_object(bid).Render_Priority = f'last{-(2+idx)}'
            self.game.get_object(bid).Render_Priority = "last-2"
            y += self.bh


class EndScreenNetworked(EndScreen):
    def __init__(self):
        MenuState.__init__(self, "EndNetworked")
        self.ranking = {}
        self.wininf = ""

    def check_state(self):
        return "Title"


class Game(ObjectHandler):
    """Game object, contains most of what's required for Azul."""

    tile_size = 30

    def __init__(self) -> None:
        """Initialize game."""
        ObjectHandler.__init__(self)
        self.keyboard = None  # Gets overwritten by Keyboard object

        self.states = {}
        self.active_state = None

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

        self.player_turn = 0

        # Tiles
        self.bag = Bag(TILECOUNT, REGTILECOUNT)

        # Cache
        self.cache = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def is_pressed(self, key) -> bool:
        """Return if key pressed."""
        return False

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

    def add_states(self, states):
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

    def get_player(self, player_id: int):
        """Get the player with player id player_id."""
        if self.players:
            return self.get_object_by_name(f"Player{player_id}")
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
            self.player_turn = random.randint(-1, self.players - 1)
        else:
            self.player_turn = "Unknown"

        cx, cy = SCREENSIZE[0] / 2, SCREENSIZE[1] / 2
        out = math.sqrt(cx**2 + cy**2) // 3 * 2

        mdeg = 360 // max_players

        for player_id in range(self.players):
            networked = False
            newp = Player(self, player_id, networked, varient_play)

            truedeg = (self.players + 1 - player_id) * (360 / self.players)
            closedeg = truedeg // mdeg * mdeg + 45
            rad = math.radians(closedeg)

            newp.location = round(cx + out * math.sin(rad)), round(
                cy + out * math.cos(rad),
            )
            self.add_object(newp)
        if self.is_host:
            self.next_turn()

        factory = Factories(self, self.factories)
        factory.location = cx, cy
        self.add_object(factory)
        self.process_objects(0)

        if self.is_host:
            self.next_turn()

    def screen_size_update(self) -> None:
        """Handle screen size updates."""
        objs_with_attr = self.get_objects_with_attr("screen_size_update")
        for oid in objs_with_attr:
            obj = self.get_object(oid)
            obj.screen_size_update()


class Keyboard:
    """Keyboard object, handles keyboard input."""

    def __init__(self, target: object, **kwargs) -> None:
        self.target = target
        self.target.keyboard = self
        self.target.is_pressed = self.is_pressed

        # Map of keyboard events to names
        self.keys: dict[int, str] = {}
        # Map of keyboard event names to functions
        self.actions: dict[str, Callable[[], None]] = {}
        # Map of names to time until function should be called again
        self.time: dict[str, float] = {}
        # Map of names to duration timer waits for function recalls
        self.delay: dict[str, float | None] = {}
        # Map of names to boolian of pressed or not
        self.active: dict[str, bool] = {}

        if kwargs:
            for name in kwargs:
                if not hasattr(kwargs[name], "__iter__"):
                    raise ValueError(
                        "Keyword arguments must be given as name=[key, self.target.function_name, delay]",
                    )
                if len(kwargs[name]) == 2:
                    key, function_name = kwargs[name]
                    delay = None
                elif len(kwargs[name]) == 3:
                    key, function_name, delay = kwargs[name]
                else:
                    raise ValueError
                self.add_listener(key, name)
                self.bind_action(name, function_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target!r})"

    def is_pressed(self, key: int) -> bool:
        """Return True if <key> is pressed."""
        return self.active.get(key, False)

    def add_listener(self, key: int, name: str) -> None:
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
            return getattr(self.target, function_name)
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

    def set_key(self, key: int, value: bool, _nochar: bool = False) -> None:
        """Set active value for key <key> to <value>."""
        if key in self.keys:
            self.set_active(self.keys[key], value)
        elif not _nochar and key < 0x110000:
            self.set_key(chr(key), value, True)

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
                    if self.delay[name] is not None:
                        self.time[name] = self.delay[name]
                    else:
                        self.time[name] = math.inf


def network_shutdown() -> None:
    pass


def run() -> None:
    ##    global game
    global SCREENSIZE
    # Set up the screen
    screen = pygame.display.set_mode(SCREENSIZE, RESIZABLE, 16)
    pygame.display.set_caption(f"{__title__} {__version__}")
    ##    pygame.display.set_icon(pygame.image.load('icon.png'))
    pygame.display.set_icon(get_tile_image(Tile(5), 32))

    # Set up the FPS clock
    clock = pygame.time.Clock()

    game = Game()
    keyboard = Keyboard(game)

    music_end = USEREVENT + 1  # This event is sent when a music track ends

    # Set music end event to our new event
    pygame.mixer.music.set_endevent(music_end)

    # Load and start playing the music
    ##    pygame.mixer.music.load('sound/')
    ##    pygame.mixer.music.play()

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

    ##    surface.lock()
    pygame.image.save(surface, os.path.join("Screenshots", filename), filename)
    ##    surface.unlock()
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
    ##    except BaseException as ex:
    ##        reraise = True#False
    ##
    ####        print('Debug: Activating Post motem.')
    ####        import pdb
    ####        pdb.post_mortem()
    ##
    ##        try:
    ##            save_crash_img()
    ##        except BaseException as svex:
    ##            print(f'Could not save crash screenshot: {", ".join(svex.args)}')
    ##        try:
    ##            import errorbox
    ##        except ImportError:
    ##            reraise = True
    ##            print(f'A {type(ex).__name__} Error Has Occored: {", ".join(ex.args)}')
    ##        else:
    ##            errorbox.errorbox('Error', f'A {type(ex).__name__} Error Has Occored: {", ".join(ex.args)}')
    ##        if reraise:
    ##            raise
    finally:
        pygame.quit()
        network_shutdown()


if __name__ == "__main__":
    cli_run()
