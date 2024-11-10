#!/usr/bin/env python3
# Azul board game clone, now on the computer!
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib

# Programmed by CoolCat467
import math
import os
import random
import time
from collections import Counter, deque
from functools import lru_cache, wraps
from pathlib import Path
from typing import Final

import pygame
from numpy import array
from pygame.locals import *

from azul.tools import (
    floor_line_subtract_generator,
    gen_random_proper_seq,
    lerp_color,
    randomize,
    saturate,
    sort_tiles,
)
from azul.Vector2 import Vector2

__title__ = "Azul"
__author__ = "CoolCat467"
__version__ = "2.0.0"

SCREENSIZE = (650, 600)
FPS = 30
# Colors
BLACK = (0, 0, 0)
BLUE = (32, 32, 255)
##BLUE    = (0, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
RED = (255, 0, 0)
MAGENTA = (255, 0, 255)
YELLOW = (255, 220, 0)
##YELLOW  = (255, 255, 0)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
ORANGE = (255, 128, 0)
DARKGREEN = (0, 128, 0)
DARKCYAN = (0, 128, 128)

if globals().get("__file__") is None:
    import importlib

    __file__ = str(
        Path(importlib.import_module("checkers.data").__path__[0]).parent
        / "game.py",
    )

ROOT_FOLDER: Final = Path(__file__).absolute().parent
DATA_FOLDER: Final = ROOT_FOLDER / "data"
FONT_FOLDER: Final = ROOT_FOLDER / "fonts"

# Game stuff
# Tiles
TILECOUNT = 100
REGTILECOUNT = 5
TILECOLORS = (BLUE, YELLOW, RED, BLACK, CYAN, (WHITE, BLUE))
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
def make_square_surf(color, size):
    """Return a surface of a square of given color and size."""
    s = int(size)
    surf = pygame.Surface((s, s))
    surf.fill(color)
    return surf


def outline_rectangle(surface, color, percent=0.1):
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


def auto_crop_clear(surface, clear=(0, 0, 0, 0)):
    """Remove unneccicary pixels from image."""
    surface = surface.convert_alpha()
    w, h = surface.get_size()
    surface.lock()
    find_end = None

    def find_end(iterfunc, rangeobj):
        for x in rangeobj:
            if not all(y == clear for y in iterfunc(x)):
                return x
        return x

    def column(x):
        return (surface.get_at((x, y)) for y in range(h))

    def row(y):
        return (surface.get_at((x, y)) for x in range(w))

    leftc = find_end(column, range(w))
    rightc = find_end(column, range(w - 1, -1, -1))
    topc = find_end(row, range(h))
    floorc = find_end(row, range(h - 1, -1, -1))
    surface.unlock()
    dim = pygame.rect.Rect(leftc, topc, rightc - leftc, floorc - topc)
    return surface.subsurface(dim)


@lru_cache
def get_tile_color(tileColor, greyshift=GREYSHIFT):
    """Return the color a given tile should be."""
    if tileColor < 0:
        if tileColor == -6:
            return GREY
        return lerp_color(TILECOLORS[abs(tileColor + 1)], GREY, greyshift)
    if tileColor < 5:
        return TILECOLORS[tileColor]
    if tileColor >= 5:
        raise ValueError(
            "Cannot properly return tile colors greater than five!",
        )
    return None


@lru_cache
def get_tile_symbol_and_color(tileColor, greyshift=GREYSHIFT):
    """Return the color a given tile should be."""
    if tileColor < 0:
        if tileColor == -6:
            return " ", GREY
        symbol, scolor = TILESYMBOLS[abs(tileColor + 1)]
        return symbol, lerp_color(scolor, GREY, greyshift)
    if tileColor <= 5:
        return TILESYMBOLS[tileColor]
    if tileColor >= 6:
        raise ValueError(
            "Cannot properly return tile colors greater than five!",
        )
    return None


def add_symbol_to_tile_surf(
    surf,
    tilecolor,
    tilesize,
    greyshift=GREYSHIFT,
    font=FONT,
):
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
    tile,
    tilesize,
    greyshift=GREYSHIFT,
    outlineSize=0.2,
    font=FONT,
):
    """Return a surface of a given tile."""
    cid = tile.color
    if cid < 5:
        color = get_tile_color(cid, greyshift)

    elif cid >= 5:
        color, outline = TILECOLORS[cid]
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


def set_alpha(surface, alpha):
    """Return a surface by replacing the alpha channel of it with given alpha value, preserve color."""
    surface = surface.copy().convert_alpha()
    w, h = surface.get_size()
    for y in range(h):
        for x in range(w):
            r, g, b = surface.get_at((x, y))[:3]
            surface.set_at((x, y), pygame.Color(r, g, b, alpha))
    return surface


@lru_cache
def get_tile_container_image(wh, back):
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
        fontName,
        fontsize=20,
        color=(0, 0, 0),
        cx=True,
        cy=True,
        antialias=False,
        background=None,
        doCache=True,
    ):
        self.font = fontName
        self.size = int(fontsize)
        self.color = color
        self.center = [cx, cy]
        self.antialias = bool(antialias)
        self.background = background
        self.doCache = bool(doCache)
        self.cache = None
        self.lastText = None
        self._changeFont()

    def __repr__(self):
        return "Font(%r, %i, %r, %r, %r, %r, %r, %r)" % (
            self.font,
            self.size,
            self.color,
            self.center[0],
            self.center[1],
            self.antialias,
            self.background,
            self.doCache,
        )

    def _changeFont(self) -> None:
        """Set self.pyfont to a new pygame.font.Font object from data we have."""
        self.pyfont = pygame.font.Font(self.font, self.size)

    def _cache(self, surface):
        """Set self.cache to surface."""
        self.cache = surface

    def get_height(self):
        """Return the height of font."""
        return self.pyfont.get_height()

    def render_nosurf(
        self,
        text,
        size=None,
        color=None,
        background=None,
        forceUpdate=False,
    ):
        """Render and return a surface of given text. Use stored data to render, if arguments change internal data and render."""
        updateCache = (
            self.cache is None or forceUpdate or text != self.lastText
        )
        # Update internal data if new values given
        if size is not None:
            self.size = int(size)
            self._changeFont()
            updateCache = True
        if color is not None:
            self.color = color
            updateCache = True
        if self.background != background:
            self.background = background
            updateCache = True

        if self.doCache:
            if updateCache:
                self.lastText = text
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
        surface,
        text,
        xy,
        size=None,
        color=None,
        background=None,
        forceUpdate=False,
    ):
        """Render given text, use stored data to render, if arguments change internal data and render."""
        surf = self.render_nosurf(text, size, color, background, forceUpdate)

        if True in self.center:
            x, y = xy
            cx, cy = self.center
            w, h = surf.get_size()
            if cx:
                x -= w / 2
            if cy:
                y -= h / 2
            xy = (int(x), int(y))

        surface.blit(surf, xy)


class ObjectHandler:
    """ObjectHandler class, meant to be used for other classes."""

    def __init__(self):
        self.objects = {}
        self.nextId = 0
        self.cache = {}

        self.recalculate_render = True
        self._render_order = ()

    def add_object(self, obj: object):
        """Add an object to the game."""
        obj.id = self.nextId
        self.objects[self.nextId] = obj
        self.nextId += 1
        self.recalculate_render = True

    def rm_object(self, obj: object):
        """Remove an object from the game."""
        del self.objects[obj.id]
        self.recalculate_render = True

    def rm_star(self):
        """Remove all objects from self.objects."""
        for oid in list(self.objects):
            self.rm_object(self.objects[oid])
        self.nextId = 0

    def get_object(self, objectId: int):
        """Return the object associated with object id given. Return None if object not found."""
        if objectId in self.objects:
            return self.objects[objectId]
        return None

    def get_objects_with_attr(self, attribute: str):
        """Return a tuple of object ids with given attribute."""
        return tuple(
            oid
            for oid in self.objects
            if hasattr(self.objects[oid], attribute)
        )

    def get_object_by_attr(self, attribute: str, value):
        """Return a tuple of object ids with <attribute> that are equal to <value>."""
        matches = []
        for oid in self.get_objects_with_attr(attribute):
            if getattr(self.objects[oid], attribute) == value:
                matches.append(oid)
        return tuple(matches)

    def get_object_given_name(self, name: str):
        """Returns a tuple of object ids with names matching <name>."""
        return self.get_object_by_attr("name", name)

    def reset_cache(self):
        """Reset the cache."""
        self.cache = {}

    def get_object_by_name(self, objName):
        """Get object by name, with cache."""
        if objName not in self.cache:
            ids = self.get_object_given_name(objName)
            if ids:
                self.cache[objName] = min(ids)
            else:
                raise RuntimeError(f"{objName} Object Not Found!")
        return self.get_object(self.cache[objName])

    def set_attr_all(self, attribute: str, value):
        """Set given attribute in all of self.objects to given value in all objects with that attribute."""
        for oid in self.get_objects_with_attr(attribute):
            setattr(self.objects[oid], attribute, value)

    def recalculate_render_order(self):
        """Recalculate the order in which to render objects to the screen."""
        new = {}
        cur = 0
        for oid in reversed(self.objects):
            obj = self.objects[oid]
            if hasattr(obj, "Render_Priority"):
                prior = obj.Render_Priority
                if isinstance(prior, str):
                    add = 0
                    if prior[:4] == "last":
                        add = prior[4:] or 0
                        try:
                            add = int(add)
                        except ValueError:
                            add = 0
                        pos = len(self.objects) + add
                    if prior[:5] == "first":
                        add = prior[5:] or 0
                        try:
                            add = int(add)
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
        new = []
        for key in sorted(revnew):
            new.append(revnew[key])
        self._render_order = tuple(new)

    def process_objects(self, time_passed: float):
        """Call the process function on all objects."""
        if self.recalculate_render:
            self.recalculate_render_order()
            self.recalculate_render = False
        for oid in iter(self.objects):
            self.objects[oid].process(time_passed)

    def render_objects(self, surface):
        """Render all objects to surface."""
        if not self._render_order or self.recalculate_render:
            self.recalculate_render_order()
            self.recalculate_render = False
        for oid in self._render_order:  # reversed(list(self.objects.keys())):
            self.objects[oid].render(surface)

    def __del__(self):
        self.reset_cache()
        self.rm_star()


class Object:
    """Object object."""

    name = "Object"

    def __init__(self, name):
        """Sets self.name to name, and other values for rendering.

        Defines the following attributes:
         self.name
         self.image
         self.location
         self.wh
         self.hidden
         self.locModOnResize
         self.id
        """
        self.name = str(name)
        self.image = None
        self.location = Vector2(
            round(SCREENSIZE[0] / 2),
            round(SCREENSIZE[1] / 2),
        )
        self.wh = 0, 0
        self.hidden = False
        self.locModOnResize = "Scale"
        self.scLast = SCREENSIZE

        self.id = 0

    def __repr__(self):
        """Return {self.name}()."""
        return f"{self.name}()"

    def getImageZero_noFix(self):
        """Return the screen location of the topleft point of self.image."""
        return (
            self.location[0] - self.wh[0] / 2,
            self.location[1] - self.wh[1] / 2,
        )

    def getImageZero(self):
        """Return the screen location of the topleft point of self.image fixed to integer values."""
        x, y = self.getImageZero_noFix()
        return int(x), int(y)

    def getRect(self):
        """Return a Rect object representing this Object's area."""
        return Rect(self.getImageZero(), self.wh)

    def point_intersects(self, screenLocation):
        """Return True if this Object intersects with a given screen location."""
        return self.getRect().collidepoint(screenLocation)

    def toImageSurfLoc(self, screenLocation):
        """Return the location a screen location would be at on the objects image. Can return invalid data."""
        # Get zero zero in image locations
        zx, zy = self.getImageZero()  # Zero x and y
        sx, sy = screenLocation  # Screen x and y
        return sx - zx, sy - zy  # Location with respect to image dimensions

    def process(self, time_passed):
        """Process Object. Replace when calling this class."""

    def render(self, surface):
        """Render self.image to surface if self.image is not None. Updates self.wh."""
        if self.image is None or self.hidden:
            return
        self.wh = self.image.get_size()
        x, y = self.getImageZero()
        surface.blit(self.image, (int(x), int(y)))

    ##        pygame.draw.rect(surface, MAGENTA, self.getRect(), 1)

    def __del__(self):
        """Delete self.image."""
        del self.image

    def screen_size_update(self):
        """Function called when screensize is changed."""
        nx, ny = self.location

        if self.locModOnResize == "Scale":
            ow, oh = self.scLast
            nw, nh = SCREENSIZE

            x, y = self.location
            nx, ny = x * (nw / ow), y * (nh / oh)

        self.location = (nx, ny)
        self.scLast = SCREENSIZE

    def get_data(self):
        """Return the data that makes this Object special."""
        data = {}
        x, y = self.location
        data["x"] = round(x)
        data["y"] = round(y)
        data["hid"] = int(self.hidden)
        data["id"] = int(self.id)
        return data

    def from_data(self, data):
        """Update an object using data."""
        self.location = float(data["x"]), float(data["y"])
        self.hidden = bool(data["hid"])
        self.id = int(data["id"])


class MultipartObject(Object, ObjectHandler):
    """Thing that is both an Object and an ObjectHandler, and is meant to be an Object made up of multiple Objects."""

    def __init__(self, name):
        """Initialize Object and ObjectHandler of self.

        Also set self._lastloc and self._lasthidden to None
        """
        Object.__init__(self, name)
        ObjectHandler.__init__(self)

        self._lastloc = None
        self._lasthidden = None

    def resetPosition(self):
        """Reset the position of all objects within."""
        raise NotImplementedError

    def getWhereTouches(self, point):
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

    def process(self, time_passed):
        """Process Object self and ObjectHandler self and call self.resetPosition on location change."""
        Object.process(self, time_passed)
        ObjectHandler.process_objects(self, time_passed)

        if self.location != self._lastloc:
            self.resetPosition()
            self._lastloc = self.location

        if self.hidden != self._lasthidden:
            self.set_attr_all("hidden", self.hidden)
            self._lasthidden = self.hidden

    def render(self, surface):
        """Render self and all parts to the surface."""
        Object.render(self, surface)
        ObjectHandler.render_objects(self, surface)

    def get_data(self):
        """Return what makes this MultipartObject special."""
        data = super().get_data()
        data["objs"] = tuple(
            [self.objects[oid].get_data() for oid in self.objects],
        )
        return data

    def from_data(self, data):
        """Update this MultipartObject from data."""
        super().from_data(self)
        for objdata in data["objs"]:
            self.objects[int(objdata["id"])].from_data(objdata)

    def __del__(self):
        Object.__del__(self)
        ObjectHandler.__del__(self)


class NerworkServer:
    """NetworkServer Class, job is to talk to connect classes over the interwebs."""

    def __init__(self, port):
        self.name = "NetworkServer"


##    def add_client(self)


class NetworkClient:
    """NetworkClient Class, job is to talk to NetworkServer and therefore other NetworkClient classes over the interwebs."""

    def __init__(self, ip_address, port):
        self.name = "NetworkClient"

    def requestData(self, dataName):
        """Request a certain field of information from the server."""


class Tile:
    """Represents a Tile."""

    def __init__(self, color):
        """Needs a color value, or this is useless."""
        self.color = color

    def __repr__(self):
        return "Tile(%i)" % self.color

    def get_data(self):
        """Return self.color."""
        return f"T[{self.color}]"

    @classmethod
    def from_data(cls, data):
        """Return a new Tile object using data."""
        return cls.__init__(int(data[2:-1]))


class TileRenderer(Object):
    """Base class for all objects that need to render tiles."""

    greyshift = GREYSHIFT
    tileSize = TILESIZE

    def __init__(
        self,
        name,
        game,
        tileSeperation="Auto",
        background=TILEDEFAULT,
    ):
        """Initialize renderer. Needs a game object for its cache and optional tile separation value and background RGB color.

        Defines the following attributes during initialization and uses throughout:
         self.game
         self.wh
         self.tileSep
         self.tileFull
         self.back
         and finally, self.imageUpdate

        The following functions are also defined:
         self.clear_image
         self.renderTile
         self.update_image (but not implemented)
         self.process
        """
        super().__init__(name)
        self.game = game

        if tileSeperation == "Auto":
            self.tileSep = self.tileSize / 3.75
        else:
            self.tileSep = tileSeperation

        self.tileFull = self.tileSize + self.tileSep
        self.back = background

        self.imageUpdate = True

    def getRect(self):
        """Return a Rect object representing this row's area."""
        wh = self.wh[0] - self.tileSep * 2, self.wh[1] - self.tileSep * 2
        location = self.location[0] - wh[0] / 2, self.location[1] - wh[1] / 2
        return Rect(location, wh)

    def clear_image(self, tileDimentions):
        """Reset self.image using tileDimentions tuple and fills with self.back. Also updates self.wh."""
        tw, th = tileDimentions
        self.wh = Vector2(
            round(tw * self.tileFull + self.tileSep),
            round(th * self.tileFull + self.tileSep),
        )
        self.image = get_tile_container_image(self.wh, self.back)

    def renderTile(self, tileObj, tileLoc):
        """Blit the surface of a given tile object onto self.image at given tile location. It is assumed that all tile locations are xy tuples."""
        x, y = tileLoc
        surf = get_tile_image(tileObj, self.tileSize, self.greyshift)
        self.image.blit(
            surf,
            (
                round(x * self.tileFull + self.tileSep),
                round(y * self.tileFull + self.tileSep),
            ),
        )

    def update_image(self):
        """Called when processing image changes, directed by self.imageUpdate being True."""
        raise NotImplementedError

    def process(self, time_passed):
        """Call self.update_image() if self.imageUpdate is True, then set self.update_image to False."""
        if self.imageUpdate:
            self.update_image()
            self.imageUpdate = False

    def get_data(self):
        """Return the data that makes this TileRenderer special."""
        data = super().get_data()
        data["tsp"] = f"{math.floor(self.tileSep*10):x}"
        data["tfl"] = f"{math.floor(self.tileFull*10):x}"
        if self.back is None:
            data["bac"] = "N"
        else:
            data["bac"] = "".join(f"{i:02x}" for i in self.back)
        return data

    def from_data(self, data):
        """Update this TileRenderer from data."""
        super().from_data(data)
        self.tileSep = int(f"0x{data['tsp']}", 16) / 10
        self.tileFull = int(f"0x{data['tfl']}", 16) / 10
        if data["bac"] == "N":
            self.back = None
        else:
            lst = [int(f"0x{data['bac'][i:i+1]}", 16) for i in range(0, 6, 2)]
            self.back = tuple(lst)


class Cursor(TileRenderer):
    """Cursor Object."""

    greyshift = GREYSHIFT
    Render_Priority = "last"

    def __init__(self, game):
        """Initialize cursor with a game it belongs to."""
        TileRenderer.__init__(self, "Cursor", game, "Auto", None)

        self.holding_number_one = False
        self.tiles = deque()

    def update_image(self):
        """Update self.image."""
        self.clear_image((len(self.tiles), 1))

        for x in range(len(self.tiles)):
            self.renderTile(self.tiles[x], (x, 0))

    def is_pressed(self):
        """Return True if the right mouse button is pressed."""
        return bool(pygame.mouse.get_pressed()[0])

    def get_held_count(self, countNumberOne=False):
        """Return the number of held tiles, can be discounting number one tile."""
        l = len(self.tiles)
        if self.holding_number_one and not countNumberOne:
            return l - 1
        return l

    def is_holding(self, countNumberOne=False):
        """Return True if the mouse is dragging something."""
        return self.get_held_count(countNumberOne) > 0

    def get_held_info(self, includeNumberOne=False):
        """Returns color of tiles are and number of tiles held."""
        if not self.is_holding(includeNumberOne):
            return None, 0
        return self.tiles[0], self.get_held_count(includeNumberOne)

    def process(self, time_passed):
        """Process cursor."""
        x, y = pygame.mouse.get_pos()
        x = saturate(x, 0, SCREENSIZE[0])
        y = saturate(y, 0, SCREENSIZE[1])
        self.location = (x, y)
        if self.imageUpdate:
            if len(self.tiles):
                self.update_image()
            else:
                self.image = None
            self.imageUpdate = False

    def force_hold(self, tiles):
        """Pretty much it's drag but with no constraints."""
        for tile in tiles:
            if tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.imageUpdate = True

    def drag(self, tiles):
        """Drag one or more tiles, as long as it's a list."""
        for tile in tiles:
            if tile is not None and tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.imageUpdate = True

    def drop(self, number="All", allowOneTile=False):
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
            self.imageUpdate = True

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


def gscBoundIndex(boundsFalureReturn=None):
    """Return a decorator for any grid or grid subclass that will keep index positions within bounds."""

    def gscBoundsKeeper(function, boundsFalureReturnValue=None):
        """Grid or Grid Subclass Decorator that keeps index positions within bounds, as long as index is first argument after self arg."""

        @wraps(function)
        def keepWithinBounds(self, index, *args, **kwargs):
            """Wrapper function that makes sure a position tuple (x, y) is valid."""
            x, y = index
            if x < 0 or x >= self.size[0]:
                return boundsFalureReturn
            if y < 0 or y >= self.size[1]:
                return boundsFalureReturn
            return function(self, index, *args, **kwargs)

        return keepWithinBounds

    @wraps(gscBoundsKeeper)
    def getGscBoundsWrapper(function):
        return gscBoundsKeeper(function, boundsFalureReturn)

    return getGscBoundsWrapper


class Grid(TileRenderer):
    """Grid object, used for boards and parts of other objects."""

    def __init__(
        self,
        size,
        game,
        tileSeperation="Auto",
        background=TILEDEFAULT,
    ):
        """Grid Objects require a size and game at least."""
        TileRenderer.__init__(self, "Grid", game, tileSeperation, background)

        self.size = tuple(size)

        self.data = array(
            [Tile(-6) for i in range(int(self.size[0] * self.size[1]))],
        ).reshape(self.size)

    def update_image(self):
        """Update self.image."""
        self.clear_image(self.size)

        for y in range(self.size[1]):
            for x in range(self.size[0]):
                self.renderTile(self.data[x, y], (x, y))

    def get_tile_point(self, screenLocation):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.point_intersects(screenLocation):
            return None
        # Otherwise, find out where screen point is in image locations
        bx, by = self.toImageSurfLoc(screenLocation)  # board x and y
        # Finally, return the full divides (no decimals) of xy location by self.tileFull.
        return int(bx // self.tileFull), int(by // self.tileFull)

    @gscBoundIndex()
    def place_tile(self, xy, tile):
        """Place a Tile Object if permitted to do so. Return True if success."""
        x, y = xy
        if self.data[x, y].color < 0:
            self.data[x, y] = tile
            del tile
            self.imageUpdate = True
            return True
        return False

    @gscBoundIndex()
    def getTile(self, xy, replace=-6):
        """Return a Tile Object from a given position in the grid if permitted. Return None on failure."""
        x, y = xy
        tileCopy = self.data[x, y]
        if tileCopy.color < 0:
            return None
        self.data[x, y] = Tile(replace)
        self.imageUpdate = True
        return tileCopy

    @gscBoundIndex()
    def get_info(self, xy):
        """Return the Tile Object at a given position without deleting it from the Grid."""
        x, y = xy
        return self.data[x, y]

    def getColors(self):
        """Return a list of the colors of tiles within self."""
        colors = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                infoTile = self.get_info((x, y))
                if infoTile.color not in colors:
                    colors.append(infoTile.color)
        return colors

    def isEmpty(self, emptyColor=-6):
        """Return True if Grid is empty (all tiles are emptyColor)."""
        colors = self.getColors()
        # Colors should only be [-6] if empty
        return colors == [emptyColor]

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

    def getColumn(self, index):
        """Return a column from self. Does not delete data from internal grid."""
        return [self.get_info((index, y)) for y in range(self.size[1])]

    def getColorsInRow(self, index, excludeNegs=True):
        """Return the colors placed in a given row in internal grid."""
        rowColors = [tile.color for tile in self.get_row(index)]
        if excludeNegs:
            rowColors = [c for c in rowColors if c >= 0]
        ccolors = Counter(rowColors)
        return sorted(ccolors.keys())

    def getColorsInColumn(self, index, excludeNegs=True):
        """Return the colors placed in a given row in internal grid."""
        columnColors = [tile.color for tile in self.getColumn(index)]
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

    @gscBoundIndex(False)
    def canPlaceTileColorAtPoint(self, position, tile):
        """Return True if tile's color is valid at given position."""
        column, row = position
        colors = set(self.getColorsInColumn(column) + self.getColorsInRow(row))
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

    @gscBoundIndex(False)
    def wall_tile_from_point(self, position):
        """Given a position, wall tile. Return success on placement. Also updates if in wall tiling mode."""
        success = False
        column, row = position
        atPoint = self.get_info(position)
        if atPoint.color <= 0 and row in self.additions:
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

    def wall_tileingMode(self, movedDict):
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

    @gscBoundIndex(([], []))
    def getTouchesContinuous(self, xy):
        """Return two lists, each of which contain all the tiles that touch the tile at given x y position, including that position."""
        rs, cs = self.size
        x, y = xy
        # Get row and column tile color data
        row = [tile.color for tile in self.get_row(y)]
        column = [tile.color for tile in self.getColumn(x)]

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
            return list(zip(one, two, strict=True))

        # Return all of the self.get_info points for each value in lst.
        def getAll(lst):
            return [self.get_info(pos) for pos in lst]

        # Get row touches
        rowTouches = comb(gt(x, rs, row), [y] * rs)
        # Get column touches
        columnTouches = comb([x] * cs, gt(y, cs, column))
        # Get real tiles from indexes and return
        return getAll(rowTouches), getAll(columnTouches)

    def scoreAdditions(self):
        """Using self.additions, which is set in self.wall_tileingMode(), return the number of points the additions scored."""
        score = 0
        for x, y in ((self.additions[y], y) for y in range(self.size[1])):
            if x is not None:
                rowt, colt = self.getTouchesContinuous((x, y))
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

    def getFilledRows(self):
        """Return the number of filled rows on this board."""
        count = 0
        for row in range(self.size[1]):
            real = (t.color >= 0 for t in self.get_row(row))
            if all(real):
                count += 1
        return count

    def hasFilledRow(self):
        """Return True if there is at least one completely filled horizontal line."""
        return self.getFilledRows() >= 1

    def getFilledColumns(self):
        """Return the number of filled rows on this board."""
        count = 0
        for column in range(self.size[0]):
            real = (t.color >= 0 for t in self.getColumn(column))
            if all(real):
                count += 1
        return count

    def getFilledColors(self):
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

    def endOfGameScoreing(self):
        """Return the additional points for this board at the end of the game."""
        score = 0
        score += self.getFilledRows() * 2
        score += self.getFilledColumns() * 7
        score += self.getFilledColors() * 10
        return score

    def process(self, time_passed):
        """Process board."""
        if self.imageUpdate and not self.variant_play:
            self.setColors(True)
        super().process(time_passed)

    def get_data(self):
        """Return data that makes this Grid Object special. Compress tile data by getting color values plus seven, then getting the hex of that as a string."""
        data = super().get_data()
        data["Wt"] = int(self.wall_tileing)
        adds = ""
        for t in self.additions.values():
            if t is None:  # If none, n
                adds += "n"
            elif isinstance(t, Tile):  # If tile, a to l
                adds += chr(t.color + 6 + 65)  # 97)
            elif isinstance(t, int):  # If int, return string repr of value.
                if t > 9:
                    raise RuntimeError(f"Integer overflow with value {t} > 9!")
                adds += str(t)
            else:
                raise RuntimeError(f'Invalid additions value "{t}"!')
        data["Ad"] = adds
        return data

    def from_data(self, data):
        """Update this Board object from data."""
        super().from_data(data)
        self.wall_tileing = bool(data["Wt"])
        for k in range(len(data["Ad"])):
            rv = data["Ad"][k]
            if rv == "n":
                v = None
            elif rv.isupper():
                v = Tile(ord(rv) - 65 - 6)
            else:
                v = int(rv)
            self.additions[k] = v


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

    def update_image(self):
        """Update self.image."""
        self.clear_image((self.size, 1))

        for x in range(len(self.tiles)):
            self.renderTile(self.tiles[x], (x, 0))

    def get_tile_point(self, screenLocation):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        xy = Grid.get_tile_point(self, screenLocation)
        if xy is None:
            return None
        x, y = xy
        return self.size - 1 - x

    def getPlaced(self):
        """Return the number of tiles in self that are not fake tiles, like grey ones."""
        return len([tile for tile in self.tiles if tile.color >= 0])

    def get_placeable(self):
        """Return the number of tiles permitted to be placed on self."""
        return self.size - self.getPlaced()

    def isFull(self):
        """Return True if this row is full."""
        return self.getPlaced() == self.size

    def get_info(self, location):
        """Return tile at location without deleting it. Return None on invalid location."""
        index = self.size - 1 - location
        if index < 0 or index > len(self.tiles):
            return None
        return self.tiles[index]

    def canPlace(self, tile):
        """Return True if permitted to place given tile object on self."""
        placeable = (tile.color == self.color) or (
            self.color < 0 and tile.color >= 0
        )
        colorCorrect = tile.color >= 0 and tile.color < 5
        numCorrect = self.get_placeable() > 0

        board = self.player.get_object_by_name("Board")
        colorNotPresent = tile.color not in board.getColorsInRow(self.size - 1)

        return placeable and colorCorrect and numCorrect and colorNotPresent

    def getTile(self, replace=-6):
        """Return the leftmost tile while deleting it from self."""
        self.tiles.appendleft(Tile(replace))
        self.imageUpdate = True
        return self.tiles.pop()

    def place_tile(self, tile):
        """Place a given Tile Object on self if permitted."""
        if self.canPlace(tile):
            self.color = tile.color
            self.tiles.append(tile)
            end = self.tiles.popleft()
            if not end.color < 0:
                raise RuntimeError(
                    "Attempted deletion of real tile from Row!",
                )
            self.imageUpdate = True

    def can_place_tiles(self, tiles):
        """Return True if permitted to place all of given tiles objects on self."""
        if len(tiles) > self.get_placeable():
            return False
        for tile in tiles:
            if not self.canPlace(tile):
                return False
        tileColors = []
        for tile in tiles:
            if tile.color not in tileColors:
                tileColors.append(tile.color)
        return not len(tileColors) > 1

    def place_tiles(self, tiles):
        """Place multiple tile objects on self if permitted."""
        if self.can_place_tiles(tiles):
            for tile in tiles:
                self.place_tile(tile)

    def wallTile(self, addToDict, blankColor=-6):
        """Move tiles around and into add dictionary for the wall tiling phase of the game. Removes tiles from self."""
        if "toBox" not in addToDict:
            addToDict["toBox"] = []
        if not self.isFull():
            addToDict[str(self.size)] = None
            return
        self.color = blankColor
        addToDict[str(self.size)] = self.getTile()
        for _i in range(self.size - 1):
            addToDict["toBox"].append(self.getTile())

    def set_background(self, color):
        """Set the background color for this row."""
        self.back = color
        self.imageUpdate = True

    def get_data(self):
        """Return the data that makes this Row special."""
        data = super().get_data()
        data["c"] = hex(self.color + 7)[2:]
        data["s"] = str(self.size)
        data["Ts"] = "".join([f"{t.color+7:x}" for t in self.tiles])
        return data

    def from_data(self, data):
        """Update this Row from data."""
        super().from_data(data)
        self.color = int(f"0x{data['c']}", 16) - 7
        self.size = int(data["s"])
        self.tiles.clear()
        for i in range(len(data["Ts"])):
            c = data["Ts"][i]
            self.tiles.append(Tile(int(f"0x{c}", 16) - 7))


class PatternLine(MultipartObject):
    """Represents multiple rows to make the pattern line."""

    size = (5, 5)

    def __init__(self, player, rowSeperation=0):
        MultipartObject.__init__(self, "PatternLine")
        self.player = player
        self.rowSep = rowSeperation

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
        self.set_attr_all("imageUpdate", True)

    def get_row(self, row):
        """Return given row."""
        return self.get_object(row)

    def resetPosition(self):
        """Reset Locations of Rows according to self.location."""
        last = self.size[1]
        w = self.get_row(last - 1).wh[0]
        if w is None:
            raise RuntimeError(
                "Image Dimensions for Row Object (row.wh) are None!",
            )
        h1 = self.get_row(0).tileFull
        h = last * h1
        self.wh = w, h
        w1 = h1 / 2

        x, y = self.location
        y -= h / 2 - w1
        for rid in self.objects:
            l = last - self.objects[rid].size
            self.objects[rid].location = x + (l * w1), y + rid * h1

    def get_tile_point(self, screenLocation):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        for y in range(self.size[1]):
            x = self.get_row(y).get_tile_point(screenLocation)
            if x is not None:
                return x, y
        return None

    def isFull(self):
        """Return True if self is full."""
        return all(self.get_row(rid).isFull() for rid in range(self.size[1]))

    def wall_tileing(self):
        """Return a dictionary to be used with wall tiling. Removes tiles from rows."""
        values = {}
        for rid in range(self.size[1]):
            self.get_row(rid).wallTile(values)
        return values

    def process(self, time_passed):
        """Process all the rows that make up the pattern line."""
        if self.hidden != self._lasthidden:
            self.set_attr_all("imageUpdate", True)
        super().process(time_passed)


class Text(Object):
    """Text object, used to render text with a given font."""

    def __init__(
        self,
        fontSize,
        color,
        background=None,
        cx=True,
        cy=True,
        name="",
    ):
        Object.__init__(self, f"Text{name}")
        self.font = Font(FONT, fontSize, color, cx, cy, True, background, True)
        self._cxy = cx, cy
        self._last = None

    def getImageZero(self):
        """Return the screen location of the topleft point of self.image."""
        x = self.location[0]
        y = self.location[1]
        if self._cxy[0]:
            x -= self.wh[0] / 2
        if self._cxy[1]:
            y -= self.wh[1] / 2
        return x, y

    def __repr__(self):
        return "<Text Object>"

    @staticmethod
    def get_font_height(font, size):
        """Return the height of font at fontsize size."""
        return pygame.font.Font(font, size).get_height()

    def update_value(self, text, size=None, color=None, background="set"):
        """Return a surface of given text rendered in FONT."""
        if background == "set":
            self.image = self.font.render_nosurf(text, size, color)
            return self.image
        self.image = self.font.render_nosurf(text, size, color, background)
        return self.image

    def getSurf(self):
        """Return self.image."""
        return self.image

    def get_tile_point(self, location):
        """Set get_tile_point attribute so that errors are not raised."""
        return

    def process(self, time_passed):
        """Process text."""
        if self.font.lastText != self._last:
            self.update_value(self.font.lastText)
            self._last = self.font.lastText

    def get_data(self):
        """Return the data that makes this Text Object special."""
        data = super().get_data()
        data["faa"] = int(self.font.antialias)

        def gethex(itera):
            return "".join(f"{i:02x}" for i in itera)

        if self.font.background is None:
            data["bg"] = "N"
        else:
            data["bg"] = gethex(self.font.background)
        data["fc"] = gethex(self.font.color)
        data["fdc"] = int(self.font.doCache)
        data["flt"] = self.font.lastText
        return data

    def from_data(self, data):
        """Update this Text Object from data."""
        super().from_data(data)
        self.font.antialias = bool(data["faa"])

        def getcolor(itera):
            return tuple(
                [int(f"0x{itera[i:i + 1]}", 16) for i in range(0, 6, 2)],
            )

        if data["bac"] == "N":
            self.font.background = None
        else:
            self.font.background = getcolor(data["bac"])
        self.font.color = getcolor(data["fc"])
        self.font.doCache = bool(data["fdc"])
        self.font.lastText = data["flt"]


class FloorLine(Row):
    """Represents a player's floor line."""

    size = 7
    number_oneColor = NUMBERONETILE

    def __init__(self, player):
        Row.__init__(self, player, self.size, background=ORANGE)
        self.name = "FloorLine"

        ##        self.font = Font(FONT, round(self.tileSize*1.2), color=BLACK, cx=False, cy=False)
        self.text = Text(round(self.tileSize * 1.2), BLACK, cx=False, cy=False)
        self.hasNumberOne = False

        gen = floor_line_subtract_generator(1)
        self.numbers = [next(gen) for i in range(self.size)]

    def __repr__(self):
        return f"FloorLine({self.player!r})"

    def render(self, surface):
        """Update self.image."""
        Row.render(self, surface)

        sx, sy = self.location
        if self.wh is None:
            return
        w, h = self.wh
        for x in range(self.size):
            xy = round(x * self.tileFull + self.tileSep + sx - w / 2), round(
                self.tileSep + sy - h / 2,
            )
            self.text.update_value(str(self.numbers[x]))
            self.text.location = xy
            self.text.render(surface)

    ##            self.font.render(surface, str(self.numbers[x]), xy)

    def place_tile(self, tile):
        """Place a given Tile Object on self if permitted."""
        self.tiles.insert(self.getPlaced(), tile)

        if tile.color == self.number_oneColor:
            self.hasNumberOne = True

        boxLid = self.player.game.get_object_by_name("BoxLid")

        def handleEnd(end):
            """Handle the end tile we are replacing. Ensures number one tile is not removed."""
            if not end.color < 0:
                if end.color == self.number_oneColor:
                    handleEnd(self.tiles.pop())
                    self.tiles.appendleft(end)
                    return
                boxLid.add_tile(end)

        handleEnd(self.tiles.pop())

        self.imageUpdate = True

    def scoreTiles(self):
        """Score self.tiles and return how to change points."""
        runningTotal = 0
        for x in range(self.size):
            if self.tiles[x].color >= 0:
                runningTotal += self.numbers[x]
            elif x < self.size - 1:
                if self.tiles[x + 1].color >= 0:
                    raise RuntimeError(
                        "Player is likely cheating! Invalid placement of FloorLine tiles!",
                    )
        return runningTotal

    def getTiles(self, emtpyColor=-6):
        """Return tuple of tiles gathered, and then either the number one tile or None."""
        tiles = []
        number_oneTile = None
        for tile in (self.tiles.pop() for i in range(len(self.tiles))):
            if tile.color == self.number_oneColor:
                number_oneTile = tile
                self.hasNumberOne = False
            elif tile.color >= 0:
                tiles.append(tile)

        for _i in range(self.size):
            self.tiles.append(Tile(emtpyColor))
        self.imageUpdate = True
        return tiles, number_oneTile

    def can_place_tiles(self, tiles):
        """Return True."""
        return True

    def get_data(self):
        """Return the data that makes this FloorLine Row special."""
        data = super().get_data()
        data["fnt"] = self.font.get_data()
        return data

    def from_data(self, data):
        """Updata this FloorLine from data."""
        super().from_data(data)
        self.font.from_data(data["fnt"])


class Factory(Grid):
    """Represents a Factory."""

    size = (2, 2)
    color = WHITE
    outline = BLUE
    outSize = 0.1

    def __init__(self, game, factoryId):
        Grid.__init__(self, self.size, game, background=None)
        self.number = factoryId
        self.name = f"Factory{self.number}"

        self.radius = math.ceil(
            self.tileFull * self.size[0] * self.size[1] / 3 + 3,
        )

    def __repr__(self):
        return "Factory(%r, %i)" % (self.game, self.number)

    def addCircle(self, surface):
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

    def render(self, surface):
        """Render Factory."""
        if not self.hidden:
            self.addCircle(surface)
        super().render(surface)

    def fill(self, tiles):
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

    def grab(self):
        """Return all tiles on this factory."""
        return [
            tile
            for tile in (
                self.getTile((x, y))
                for x in range(self.size[0])
                for y in range(self.size[1])
            )
            if tile.color != -6
        ]

    def grabColor(self, color):
        """Return all tiles of color given in the first list, and all non-matches in the second list."""
        tiles = self.grab()
        right, wrong = [], []
        for tile in tiles:
            if tile.color == color:
                right.append(tile)
            else:
                wrong.append(tile)
        return right, wrong

    def process(self, time_passed):
        """Process self."""
        if self.imageUpdate:
            self.radius = self.tileFull * self.size[0] * self.size[1] / 3 + 3
        super().process(time_passed)

    def get_data(self):
        """Return what makes this Factory Grid special."""
        data = super().get_data()
        data["n"] = self.number
        data["r"] = f"{math.ceil(self.radius):x}"
        return data

    def from_data(self, data):
        """Update this Factory from data."""
        super().from_data(data)
        self.number = int(data["n"])
        self.name = f"Factory{self.number}"
        self.radius = int(f"0x{data['r']}", 16)


class Factories(MultipartObject):
    """Factories Multipart Object, made of multiple Factory Objects."""

    teach = 4

    def __init__(self, game, factories: int, size="Auto"):
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

        self.divyUpTiles()

    def __repr__(self):
        return "Factories(%r, %i, ...)" % (self.game, self.count)

    def resetPosition(self):
        """Reset the position of all factories within."""
        degrees = 360 / self.count
        for i in range(self.count):
            rot = math.radians(degrees * i)
            self.objects[i].location = (
                math.sin(rot) * self.size + self.location[0],
                math.cos(rot) * self.size + self.location[1],
            )

    def process(self, time_passed):
        """Process factories. Does not react to cursor if hidden."""
        super().process(time_passed)
        if not self.hidden:
            cursor = self.game.get_object_by_name("Cursor")
            if cursor.is_pressed() and not cursor.is_holding():
                obj, point = self.getWhereTouches(cursor.location)
                if obj is not None and point is not None:
                    oid = int(obj[7:])
                    tileAtPoint = self.objects[oid].get_info(point)
                    if (tileAtPoint is not None) and tileAtPoint.color >= 0:
                        table = self.game.get_object_by_name("TableCenter")
                        select, tocenter = self.objects[oid].grabColor(
                            tileAtPoint.color,
                        )
                        if tocenter:
                            table.add_tiles(tocenter)
                        cursor.drag(select)

    def divyUpTiles(self, emptyColor=-6):
        """Divy up tiles to each factory from the bag."""
        # For every factory we have,
        for fid in range(self.count):
            # Draw tiles for the factory
            drawn = []
            for _i in range(self.teach):
                # If the bag is not empty,
                if not self.game.bag.isEmpty():
                    # Draw a tile from the bag.
                    drawn.append(self.game.bag.draw_tile())
                else:  # Otherwise, get the box lid
                    boxLid = self.game.get_object_by_name("BoxLid")
                    # If the box lid is not empty,
                    if not boxLid.isEmpty():
                        # Add all the tiles from the box lid to the bag
                        self.game.bag.add_tiles(boxLid.getTiles())
                        # and shake the bag to randomize everything
                        self.game.bag.reset()
                        # Then, grab a tile from the bag like usual.
                        drawn.append(self.game.bag.draw_tile())
                    else:
                        # "In the rare case that you run out of tiles again
                        # while there are none left in the lid, start a new
                        # round as usual even though are not all factory
                        # displays are properly filled."
                        drawn.append(Tile(emptyColor))
            # Place drawn tiles on factory
            self.objects[fid].fill(drawn)

    def is_all_empty(self):
        """Return True if all factories are empty."""
        return all(self.objects[fid].isEmpty() for fid in range(self.count))

    def get_data(self):
        """Return what makes this Factories ObjectHandler special."""
        data = super().get_data()
        data["cnt"] = f"{self.count:x}"
        data["sz"] = f"{math.ceil(self.size):x}"
        return data

    def from_data(self, data):
        """Update these Factories with data."""
        super().from_data(data)
        self.count = int(f"0x{data['cnt']}", 16)
        self.size = int(f"0x{data['sz']}", 16)


class TableCenter(Grid):
    """Object that represents the center of the table."""

    size = (6, 6)
    firstTileColor = NUMBERONETILE

    def __init__(self, game, hasOne=True):
        """Requires a game object handler to exist in."""
        Grid.__init__(self, self.size, game, background=None)
        self.game = game
        self.name = "TableCenter"

        self.firstTileExists = False
        if hasOne:
            self.add_number_one_tile()

        self.nextPosition = (0, 0)

    def __repr__(self):
        return f"TableCenter({self.game!r})"

    def add_number_one_tile(self):
        """Add the number one tile to the internal grid."""
        if not self.firstTileExists:
            x, y = self.size
            self.place_tile((x - 1, y - 1), Tile(self.firstTileColor))
            self.firstTileExists = True

    def add_tile(self, tile):
        """Add a Tile Object to the Table Center Grid."""
        self.place_tile(self.nextPosition, tile)
        x, y = self.nextPosition
        x += 1
        y += int(x // self.size[0])
        x %= self.size[0]
        y %= self.size[1]
        self.nextPosition = (x, y)
        self.imageUpdate = True

    def add_tiles(self, tiles, sort=True):
        """Add multiple Tile Objects to the Table Center Grid."""
        for tile in tiles:
            self.add_tile(tile)
        if sort and tiles:
            self.reorder_tiles()

    def reorder_tiles(self, replace=-6):
        """Re-organize tiles by Color."""
        full = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if self.firstTileExists:
                    if self.get_info((x, y)).color == self.firstTileColor:
                        continue
                at = self.getTile((x, y), replace)

                if at is not None:
                    full.append(at)
        sortedTiles = sorted(full, key=sort_tiles)
        self.nextPosition = (0, 0)
        self.add_tiles(sortedTiles, False)

    def pull_tiles(self, tileColor, replace=-6):
        """Remove all of the tiles of tileColor from the Table Center Grid."""
        toPull = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                infoTile = self.get_info((x, y))
                if infoTile.color == tileColor:
                    toPull.append((x, y))
                elif self.firstTileExists:
                    if infoTile.color == self.firstTileColor:
                        toPull.append((x, y))
                        self.firstTileExists = False
        tiles = [self.getTile(pos, replace) for pos in toPull]
        self.reorder_tiles(replace)
        return tiles

    def process(self, time_passed):
        """Process factories."""
        if not self.hidden:
            cursor = self.game.get_object_by_name("Cursor")
            if (
                cursor.is_pressed()
                and not cursor.is_holding()
                and not self.isEmpty()
            ):
                if self.point_intersects(cursor.location):
                    point = self.get_tile_point(cursor.location)
                    # Shouldn't return none anymore since we have point_intersects now.
                    colorAtPoint = self.get_info(point).color
                    if colorAtPoint >= 0 and colorAtPoint < 5:
                        cursor.drag(self.pull_tiles(colorAtPoint))
        super().process(time_passed)

    def get_data(self):
        """Return what makes the TableCenter special."""
        data = super().get_data()
        data["fte"] = int(self.firstTileExists)
        x, y = self.nextPosition
        data["np"] = f"{x}{y}"
        return data

    def from_data(self, data):
        """Update the TableCenter from data."""
        super().from_data(data)
        self.firstTileExists = bool(data["fte"])
        x, y = data["np"]
        self.nextPosition = int(x), int(y)


class Bag:
    """Represents the bag full of tiles."""

    def __init__(self, numTiles=100, tileTypes=5):
        self.numTiles = int(numTiles)
        self.tileTypes = int(tileTypes)
        self.tileNames = [chr(65 + i) for i in range(self.tileTypes)]
        self.percentEach = (self.numTiles / self.tileTypes) / 100
        self.full_reset()

    def full_reset(self):
        """Reset the bag to a full, re-randomized bag."""
        self.tiles = deque(
            gen_random_proper_seq(
                self.numTiles,
                **{tileName: self.percentEach for tileName in self.tileNames},
            ),
        )

    def __repr__(self):
        return "Bag(%i, %i)" % (self.numTiles, self.tileTypes)

    def reset(self):
        """Randomize all the tiles in the bag."""
        self.tiles = deque(randomize(self.tiles))

    def get_color(self, tileName):
        """Return the color of a named tile."""
        if tileName not in self.tileNames:
            raise ValueError(f"Tile Name {tileName} Not Found!")
        return self.tileNames.index(tileName)

    def get_tile(self, tileName):
        """Return a Tile Object from a tile name."""
        return Tile(self.get_color(tileName))

    def getCount(self):
        """Return number of tiles currently held."""
        return len(self.tiles)

    def isEmpty(self):
        """Return True if no tiles are currently held."""
        return self.getCount() == 0

    def draw_tile(self):
        """Return a random Tile Object from the bag. Return None if no tiles to draw."""
        if not self.isEmpty():
            return self.get_tile(self.tiles.pop())
        return None

    def get_name(self, tileColor):
        """Return the name of a tile given it's color."""
        try:
            return self.tileNames[tileColor]
        except IndexError:
            raise ValueError("Invalid Tile Color!")

    def add_tile(self, tileObject):
        """Add a Tile Object to the bag."""
        name = self.get_name(int(tileObject.color))
        rnge = (0, len(self.tiles) - 1)
        if rnge[1] - rnge[0] <= 1:
            index = 0
        else:
            index = random.randint(rnge[0], rnge[1])
        ##        self.tiles.insert(random.randint(0, len(self.tiles)-1), self.get_name(int(tileObject.color)))
        self.tiles.insert(index, name)
        del tileObject

    def add_tiles(self, tileObjects):
        """Add multiple Tile Objects to the bag."""
        for tileObject in tileObjects:
            self.add_tile(tileObject)


class BoxLid(Object):
    """BoxLid Object, represents the box lid were tiles go before being added to the bag again."""

    def __init__(self, game):
        Object.__init__(self, "BoxLid")
        self.game = game
        self.tiles = deque()

    def __repr__(self):
        return f"BoxLid({self.game!r})"

    def add_tile(self, tile):
        """Add a tile to self."""
        if tile.color >= 0 and tile.color < 5:
            self.tiles.append(tile)
        else:
            raise Warning(
                f"BoxLid.add_tile tried to add an invalid tile to self ({tile.color}). Be careful, bad things might be trying to happen.",
            )

    def add_tiles(self, tiles):
        """Add multiple tiles to self."""
        for tile in tiles:
            self.add_tile(tile)

    def getTiles(self):
        """Return all tiles in self while deleting them from self."""
        return [self.tiles.popleft() for i in range(len(self.tiles))]

    def isEmpty(self):
        """Return True if self is empty (no tiles on it)."""
        return len(self.tiles) == 0

    def get_data(self):
        """Return what makes this BoxLid Object special."""
        data = super().get_data()
        data["Ts"] = "".join(f"{t.color+7:x}" for t in self.tiles)
        return data

    def from_data(self, data):
        """Update this BoxLid from data."""
        super().from_data(data)
        self.tiles.clear()
        self.add_tiles(Tile(int(f"0x{t}", 16) - 7) for t in data["Ts"])


class Player(MultipartObject):
    """Represents a player. Made of lots of objects."""

    def __init__(
        self,
        game,
        playerId: int,
        networked=False,
        varient_play=False,
    ):
        """Requires a player Id and can be told to be controlled by the network or be in variant play mode."""
        MultipartObject.__init__(self, "Player%i" % playerId)

        self.game = game
        self.pid = playerId
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

        self.updateScore()

        self._lastloc = 0, 0

    def __repr__(self):
        return "Player(%r, %i, %s, %s)" % (
            self.game,
            self.pid,
            self.networked,
            self.varient_play,
        )

    def updateScore(self):
        """Update the scorebox for this player."""
        scoreBox = self.get_object_by_name("Text")
        scoreBox.update_value(f"Player {self.pid+1}: {self.score}")

    def turnNow(self):
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

    def end_of_turn(self):
        """It is no longer this player's turn."""
        if self.is_turn:
            pattern_line = self.get_object_by_name("PatternLine")
            pattern_line.set_background(None)
        self.is_turn = False

    def itsTheEnd(self):
        """Function called by end state when game is over; Hide pattern lines and floor line."""
        pattern = self.get_object_by_name("PatternLine")
        floor = self.get_object_by_name("FloorLine")

        pattern.hidden = True
        floor.hidden = True

    def resetPosition(self):
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

    def wall_tileing(self):
        """Do the wall tiling phase of the game for this player."""
        self.is_wall_tileing = True
        pattern_line = self.get_object_by_name("PatternLine")
        self.get_object_by_name("FloorLine")
        board = self.get_object_by_name("Board")
        boxLid = self.game.get_object_by_name("BoxLid")

        data = pattern_line.wall_tileing()
        boxLid.add_tiles(data["toBox"])
        del data["toBox"]

        board.wall_tileingMode(data)

    def done_wall_tileing(self):
        """Return True if internal Board is done wall tiling."""
        board = self.get_object_by_name("Board")
        return not board.is_wall_tileing()

    def next_round(self):
        """Called when player is done wall tiling."""
        self.is_wall_tileing = False

    def scorePhase(self):
        """Do the scoring phase of the game for this player."""
        board = self.get_object_by_name("Board")
        floorLine = self.get_object_by_name("FloorLine")
        boxLid = self.game.get_object_by_name("BoxLid")

        def saturatescore():
            if self.score < 0:
                self.score = 0

        self.score += board.scoreAdditions()
        self.score += floorLine.scoreTiles()
        saturatescore()

        toBox, number_one = floorLine.getTiles()
        boxLid.add_tiles(toBox)

        self.updateScore()

        return number_one

    def endOfGameScoring(self):
        """Update final score with additional end of game points."""
        board = self.get_object_by_name("Board")

        self.score += board.endOfGameScoreing()

        self.updateScore()

    def hasHorizLine(self):
        """Return True if this player has a horizontal line on their game board filled."""
        board = self.get_object_by_name("Board")
        return board.hasFilledRow()

    def getHorizontalLines(self):
        """Return the number of filled horizontal lines this player has on their game board."""
        board = self.get_object_by_name("Board")
        return board.getFilledRows()

    def process(self, time_passed):
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
                    obj, point = self.getWhereTouches(cursor.location)
                    if (
                        obj is not None and point is not None
                    ):  # Something pressed
                        if cursor.is_holding():  # Cursor holding tiles
                            madeMove = False
                            if not self.is_wall_tileing:  # Is wall tiling:
                                if obj == "PatternLine":
                                    pos, rowNum = point
                                    row = pattern_line.get_row(rowNum)
                                    if not row.isFull():
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
                                                madeMove = True
                                            else:
                                                cursor.force_hold(tiles)
                                elif obj == "FloorLine":
                                    tiles_to_add = cursor.drop()
                                    if floorLine.isFull():  # Floor is full,
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
                                    madeMove = True
                            elif (
                                not self.just_held
                            ):  # Cursor holding and wall tiling
                                if obj == "Board":
                                    atPoint = board.get_info(point)
                                    if atPoint.color == -6:
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

                            if madeMove:
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
                                        columnNum, rowNum = point
                                        tile = (
                                            board.get_tile_for_cursor_by_row(
                                                rowNum,
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

    def get_data(self):
        """Return what makes this Player MultipartObject special."""
        data = super().get_data()
        data["pi"] = int(self.pid)
        data["sc"] = f"{self.score:x}"
        data["tu"] = int(self.is_turn)
        data["iwt"] = int(self.is_wall_tileing)
        return data

    def from_data(self, data):
        """Update this Player from data."""
        super().from_data()
        self.pid = int(data["pi"])
        self.score = int(data["sc"], 16)
        self.is_turn = bool(data["tu"])
        self.is_wall_tileing = bool(data["iwt"])


class Button(Text):
    """Button Object."""

    textcolor = BUTTONTEXTCOLOR
    backcolor = BUTTONBACKCOLOR

    def __init__(
        self,
        state,
        name,
        minSize=10,
        initValue="",
        fontSize=BUTTONFONTSIZE,
    ):
        super().__init__(fontSize, self.textcolor, background=None)
        self.name = f"Button{name}"
        self.state = state

        self.minsize = int(minSize)
        self.update_value(initValue)

        self.borderWidth = math.floor(fontSize / 12)  # 5

        self.action = lambda: None
        self.delay = 0.6
        self.cur_time = 1

    def __repr__(self):
        return f"Button({self.name[6:]}, {self.state}, {self.minsize}, {self.font.lastText}, {self.font.pyfont})"

    def get_height(self):
        return self.font.get_height()

    def bind_action(self, function):
        """When self is pressed, call given function exactly once. Function takes no arguments."""
        self.action = function

    def update_value(self, text, size=None, color=None, background="set"):
        disp = str(text).center(self.minsize)
        super().update_value(f" {disp} ", size, color, background)
        self.font.lastText = disp

    def render(self, surface):
        if not self.hidden:
            text_rect = self.getRect()
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

    def is_pressed(self):
        """Return True if this button is pressed."""
        cursor = self.state.game.get_object_by_name("Cursor")
        if not self.hidden and cursor.is_pressed():
            if self.point_intersects(cursor.location):
                return True
        return False

    def process(self, time_passed):
        """Call self.action one time when pressed, then wait self.delay to call again."""
        if self.cur_time > 0:
            self.cur_time = max(self.cur_time - time_passed, 0)
        else:
            if self.is_pressed():
                self.action()
                self.cur_time = self.delay
        if self.font.lastText != self._last:
            self.textSize = self.font.pyfont.size(f" {self.font.lastText} ")
        super().process(time_passed)

    def from_data(self, data):
        """Update this Button from data."""
        super().from_data(data)
        self.update_value(data["flt"])


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

    buttonMin = 10
    fontsize = BUTTONFONTSIZE

    def __init__(self, name):
        """Initialize GameState and set up self.bh."""
        super().__init__(name)
        self.bh = Text.get_font_height(FONT, self.fontsize)

        self.toState = None

    def add_button(
        self,
        name,
        value,
        action,
        location="Center",
        size=fontsize,
        minlen=buttonMin,
    ):
        """Add a new Button object to self.game with arguments. Return button id."""
        button = Button(self, name, minlen, value, size)
        button.bind_action(action)
        if location != "Center":
            button.location = location
        self.game.add_object(button)
        return button.id

    def add_text(
        self,
        name,
        value,
        location,
        color=BUTTONTEXTCOLOR,
        cx=True,
        cy=True,
        size=fontsize,
    ):
        """Add a new Text object to self.game with arguments. Return text id."""
        text = Text(size, color, None, cx, cy, name)
        text.location = location
        text.update_value(value)
        self.game.add_object(text)
        return text.id

    def entry_actions(self):
        """Clear all objects, add cursor object, and set up toState."""
        self.toState = None

        self.game.rm_star()
        self.game.add_object(Cursor(self.game))

    def set_var(self, attribute, value):
        """Set MenuState.{attribute} to {value}."""
        setattr(self, attribute, value)

    def to_state(self, stateName):
        """Return a function that will change game state to stateName."""

        def toStateName():
            f"Set MenuState.toState to {stateName}."
            self.toState = stateName

        return toStateName

    def var_dependant_to_state(self, **kwargs):
        """Attribute name = (target value, on trigger tostate)."""
        for state in kwargs:
            if not len(kwargs[state]) == 2:
                raise ValueError(f'Key "{state}" is invalid!')
            key, value = kwargs[state]
            if not hasattr(self, key):
                raise ValueError(
                    f'{self} object does not have attribute "{key}"!',
                )

        def to_state_by_attributes():
            """Set MenuState.toState to a new state if conditions are right."""
            for state in kwargs:
                key, value = kwargs[state]
                if getattr(self, key) == value:
                    self.toState = state

        return to_state_by_attributes

    def with_update(self, update_function):
        """Return a wrapper for a function that will call update_function after function."""

        def update_wrapper(function):
            """Wrapper for any function that could require a screen update."""

            @wraps(function)
            def function_with_update():
                """Call main function, then update function."""
                function()
                update_function()

            return function_with_update

        return update_wrapper

    def update_text(self, textName, valueFunc):
        """Update text object with textName's display value."""

        def updater():
            f"Update text object {textName}'s value with {valueFunc}."
            text = self.game.get_object_by_name(f"Text{textName}")
            text.update_value(valueFunc())

        return updater

    def toggle_button_state(self, textname, boolattr, textfunc):
        """Return function that will toggle the value of text object <textname>, toggling attribute <boolattr>, and setting text value with textfunc."""

        def valfunc():
            """Return the new value for the text object. Gets called AFTER value is toggled."""
            return textfunc(getattr(self, boolattr))

        @self.with_update(self.update_text(textname, valfunc))
        def toggleValue():
            """Toggle the value of boolattr."""
            self.set_var(boolattr, not getattr(self, boolattr))

        return toggleValue

    def check_state(self):
        """Return self.toState."""
        return self.toState


class InitState(GameState):
    def __init__(self):
        super().__init__("Init")

    def entry_actions(self):
        self.game.keyboard.add_listener("\x7f", "Delete")
        self.game.keyboard.bind_action("Delete", "screenshot", 5)

        self.game.keyboard.add_listener("\x1b", "Escape")
        self.game.keyboard.bind_action("Escape", "raise_close", 5)

        self.game.keyboard.add_listener("0", "Debug")
        self.game.keyboard.bind_action("Debug", "debug", 5)

    def check_state(self):
        return "Title"


class TitleScreen(MenuState):
    """Game state when the title screen is up."""

    def __init__(self):
        super().__init__("Title")

    def entry_actions(self):
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

    def __init__(self):
        super().__init__("Credits")

    def entry_actions(self):
        super().entry_actions()

    def check_state(self):
        return "Title"


class SettingsScreen(MenuState):
    """Game state when user is defining game type, players, etc."""

    def __init__(self):
        super().__init__("Settings")

        self.player_count = 0  # 2
        self.host_mode = True
        self.variant_play = False

    def entry_actions(self):
        """Add cursor object and tons of button and text objects to the game."""
        super().entry_actions()

        def add_numbers(start, end, widthEach, cx, cy):
            count = end - start + 1
            evencount = count % 2 == 0
            mid = count // 2

            def addNumber(number, display, cx, cy):
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
                def set_player_count():
                    f"Set variable player_count to {display} while updating text."
                    return self.set_var("player_count", display)

                self.add_button(
                    f"SetCount{number}",
                    str(display),
                    set_player_count,
                    (cx + (widthEach * x), cy),
                    size=self.fontsize / 1.5,
                    minlen=3,
                )

            for i in range(count):
                addNumber(i, start + i, cx, cy)

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

        def varient_text(x):
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

    def exit_actions(self):
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
            and table.isEmpty()
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
        for pid in range(self.game.players):
            # Activate wall tiling mode.
            player = self.game.get_player(pid)
            player.wall_tileing()
            # Add that player's pid to the list of not-processed players.
            self.not_processed.append(player.pid)

        # Start processing players.
        self.game.next_turn()

    def do_actions(self):
        if self.not_processed:
            if self.game.player_turn in self.not_processed:
                player = self.game.get_player(self.game.player_turn)
                if player.done_wall_tileing():
                    # Once player is done wall tiling, score their moves.
                    number_one = (
                        player.scorePhase()
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
            self.game.get_player(pid) for pid in range(self.game.players)
        )
        complete = (player.hasHorizLine() for player in players)
        self.newRound = not any(complete)

    def do_actions(self):
        if self.newRound:
            fact = self.game.get_object_by_name("Factories")
            # This also handles bag re-filling from box lid.
            fact.divyUpTiles()

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
        for pid in range(self.game.players):
            player = self.game.get_player(pid)
            player.itsTheEnd()
            if player.score not in scpid:
                scpid[player.score] = [pid]
            else:
                scpid[player.score] += [pid]
        # make sure no ties and establish rank
        rank = 1
        for score in sorted(scpid, reverse=True):
            pids = scpid[score]
            if len(pids) > 1:
                # If players have same score,
                # most horizontal lines is tie breaker.
                players = [self.game.get_player(pid) for pid in pids]
                lines = [(p.getHorizontalLines(), p.pid) for p in players]
                last = None
                for c, pid in sorted(lines, key=lambda x: x[0], reverse=True):
                    if last == c:
                        self.ranking[rank - 1] += [pid + 1]
                        continue
                    last = c
                    self.ranking[rank] = [pid + 1]
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
        idx = 0
        for line in self.wininf.split("\n"):
            self.add_text(f"Line{idx}", line, (x, y), cx=True, cy=False)
            ##            self.game.get_object(bid).Render_Priority = f'last{-(2+idx)}'
            self.game.get_object(bid).Render_Priority = "last-2"
            idx += 1
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

    tileSize = 30

    def __init__(self):
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

    def __repr__(self):
        return "Game()"

    def is_pressed(self, key):
        """Function that is meant to be overwritten by the Keyboard object."""
        return False

    def debug(self):
        """Debug."""

    def screenshot(self):
        """Save a screenshot of this game's most recent frame."""
        surface = pygame.surface.Surface(SCREENSIZE)
        self.render(surface)
        strTime = "-".join(time.asctime().split(" "))
        filename = f"Screenshot_at_{strTime}.png"

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

    def raise_close(self):
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

    def set_state(self, new_state_name):
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

    def update_state(self):
        """Perform the actions of the active state and potentially change states."""
        # Only continue if there is an active state
        if self.active_state is None:
            return

        # Perform the actions of the active state and check conditions
        self.active_state.do_actions()

        new_state_name = self.active_state.check_state()
        if new_state_name is not None:
            self.set_state(new_state_name)

    def add_object(self, obj):
        """Add an object to the game."""
        obj.game = self
        super().add_object(obj)

    def render(self, surface):
        """Render all of self.objects to the screen."""
        surface.fill(self.background_color)
        self.render_objects(surface)

    def process(self, time_passed):
        """Process all the objects and self."""
        if not self.initialized_state and self.keyboard is not None:
            self.set_state("Init")
            self.initialized_state = True
        self.process_objects(time_passed)
        self.update_state()

    def get_player(self, pid):
        """Get the player with player id pid."""
        if self.players:
            return self.get_object_by_name(f"Player{pid}")
        raise RuntimeError("No players!")

    def player_turn_over(self):
        """Call end_of_turn for current player."""
        if self.player_turn >= 0 and self.player_turn < self.players:
            old_player = self.get_player(self.player_turn)
            if old_player.is_turn:
                old_player.end_of_turn()

    def next_turn(self):
        """Tell current player it's the end of their turn, and update who's turn it is and now it's their turn."""
        if self.is_host:
            self.player_turn_over()
            last = self.player_turn
            self.player_turn = (self.player_turn + 1) % self.players
            if self.player_turn == last and self.players > 1:
                self.next_turn()
                return
            new_player = self.get_player(self.player_turn)
            new_player.turnNow()

    def start_game(
        self,
        players,
        varient_play=False,
        host_mode=True,
        address="",
    ):
        """Start a new game."""
        self.reset_cache()
        maxPlayers = 4
        self.players = saturate(players, 1, maxPlayers)
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

        mdeg = 360 // maxPlayers

        for pid in range(self.players):
            networked = False
            newp = Player(self, pid, networked, varient_play)

            truedeg = (self.players + 1 - pid) * (360 / self.players)
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

    def screen_size_update(self):
        """Function called when screen size is updated."""
        objs_with_attr = self.get_objects_with_attr("screen_size_update")
        for oid in objs_with_attr:
            obj = self.get_object(oid)
            obj.screen_size_update()


class Keyboard:
    """Keyboard object, handles keyboard input."""

    def __init__(self, target, **kwargs):
        self.target = target
        self.target.keyboard = self
        self.target.is_pressed = self.is_pressed

        self.keys = {}  # Map of keyboard events to names
        self.actions = {}  # Map of keyboard event names to functions
        self.time = (
            {}
        )  # Map of names to time until function should be called again
        self.delay = (
            {}
        )  # Map of names to duration timer waits for function recalls
        self.active = {}  # Map of names to boolian of pressed or not

        if kwargs:
            for name in kwargs:
                if not hasattr(kwargs[name], "__iter__"):
                    raise ValueError(
                        "Keyword arguments must be given as name=[key, self.target.functionName, delay]",
                    )
                if len(kwargs[name]) == 2:
                    key, functionName = kwargs[name]
                    delay = None
                elif len(kwargs[name]) == 3:
                    key, functionName, delay = kwargs[name]
                else:
                    raise ValueError
                self.add_listener(key, name)
                self.bind_action(name, functionName)

    def __repr__(self):
        return f"Keyboard({self.target!r})"

    def is_pressed(self, key):
        """Return True if <key> is pressed."""
        if key in self.active:
            return self.active[key]
        return False

    def add_listener(self, key: int, name: str):
        """Listen for key down events with event.key == key argument and when that happens set self.actions[name] to true."""
        self.keys[key] = name  # key to name
        self.actions[name] = lambda: None  # name to function
        self.time[name] = 0  # name to time until function recall
        self.delay[name] = None  # name to function recall delay
        self.active[name] = False  # name to boolian of pressed

    def get_function_from_target(self, functionName: str):
        """Return function with name functionName from self.target."""
        if hasattr(self.target, functionName):
            return getattr(self.target, functionName)
        return lambda: None

    def bind_action(self, name: str, targetFunctionName: str, delay=None):
        """Bind an event we are listening for to calling a function, can call multiple times if delay is not None."""
        self.actions[name] = self.get_function_from_target(targetFunctionName)
        self.delay[name] = delay

    def set_active(self, name: str, value: bool):
        """Set active value for key name <name> to <value>."""
        if name in self.active:
            self.active[name] = bool(value)
            if not value:
                self.time[name] = 0

    def set_key(self, key: int, value: bool, _nochar=False):
        """Set active value for key <key> to <value>."""
        if key in self.keys:
            self.set_active(self.keys[key], value)
        elif not _nochar and key < 0x110000:
            self.set_key(chr(key), value, True)

    def read_event(self, event):
        """Handles an event."""
        if event.type == KEYDOWN:
            self.set_key(event.key, True)
        elif event.type == KEYUP:
            self.set_key(event.key, False)

    def read_events(self, events):
        """Handles a list of events."""
        for event in events:
            self.read_event(event)

    def process(self, time_passed):
        """Sends commands to self.target based on pressed keys and time."""
        for name in self.active:
            if self.active[name]:
                self.time[name] = max(self.time[name] - time_passed, 0)
                if self.time[name] == 0:
                    self.actions[name]()
                    if self.delay[name] is not None:
                        self.time[name] = self.delay[name]
                    else:
                        self.time[name] = math.inf


def network_shutdown():
    pass


def run():
    global game
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

    MUSIC_END = USEREVENT + 1  # This event is sent when a music track ends

    # Set music end event to our new event
    pygame.mixer.music.set_endevent(MUSIC_END)

    # Load and start playing the music
    ##    pygame.mixer.music.load('sound/')
    ##    pygame.mixer.music.play()

    RUNNING = True

    # While the game is active
    while RUNNING:
        # Event handler
        for event in pygame.event.get():
            if event.type == QUIT:
                RUNNING = False
            elif event.type == MUSIC_END:
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


def save_crash_img():
    """Save the last frame before the game crashed."""
    surface = pygame.display.get_surface().copy()
    strTime = "-".join(time.asctime().split(" "))
    filename = f"Crash_at_{strTime}.png"

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
