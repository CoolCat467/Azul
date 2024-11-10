#!/usr/bin/env python3
# Azul board game clone, now on the computer!
# -*- coding: utf-8 -*-

# Programmed by CoolCat467

from __future__ import annotations

from typing import Any, Iterable, Callable, Generator, Sequence, Awaitable

import os
import random
import time
import math
from collections import deque, Counter
from functools import wraps, lru_cache

from numpy import array

from pygame.locals import *
import pygame
from pygame.rect import Rect
from pygame.surface import Surface
import trio

from async_clock import Clock
from component import Event, ComponentManager, Component
import crop
import sprite


__title__ = 'Azul'
__author__ = 'CoolCat467'
__version__ = '0.0.0'
__ver_major__ = 0
__ver_minor__ = 0
__ver_patch__ = 0

SCREENSIZE = (650, 600)
FPS = 30
VSYNC = True

# Colors
BLACK   = (0, 0, 0)
BLUE    = (32, 32, 255)
##BLUE    = (0, 0, 255)
GREEN   = (0, 255, 0)
CYAN    = (0, 255, 255)
RED     = (255, 0, 0)
MAGENTA = (255, 0, 255)
YELLOW  = (255, 220, 0)
##YELLOW  = (255, 255, 0)
WHITE   = (255, 255, 255)
GREY    = (170, 170, 170)
ORANGE  = (255, 128, 0)
DARKGREEN=(0, 128, 0)
DARKCYAN= (0, 128, 128)

# Client stuff
# Tiles
TILECOUNT = 100
REGTILECOUNT = 5
tile_colorS = (BLUE, YELLOW, RED, BLACK, CYAN, (WHITE, BLUE))
TILESYMBOLS= (('*', WHITE), ('X', BLACK), ('+', BLACK), ('?', YELLOW), ('&', ORANGE), ('1', BLUE))
NUMBERONETILE = 5
TILESIZE = 15

# Colors
BACKGROUND = (0, 192, 16)
TILEDEFAULT = ORANGE
SCORECOLOR = BLACK
PATSELECTCOLOR = DARKGREEN
BUTTONTEXTCOLOR = DARKCYAN
BUTTONBACKCOLOR = WHITE
GREYSHIFT = 0.75#0.65

# Font
FONT = 'fonts/RuneScape-UF-Regular.ttf'
SCOREFONTSIZE = 30
BUTTONFONTSIZE = 60

from tools import *
from Vector2 import *


class Tile:
    __slots__ = ('color',)
    "Represents a Tile."
    def __init__(self, color: int) -> None:
        "Needs a color value, or this is useless."
        self.color = color

    def __repr__(self) -> str:
        return f'Tile({self.color})'

    def get_data(self) -> str:
        "Return self.color"
        return f'T[{self.color}]'


def make_square_surf(color: tuple[int, int, int], size: int) -> Surface:
    "Return a surface of a square of given color and size."
    surf = pygame.Surface((size, size))
    surf.fill(color)
    return surf

def outline_rectangle(surface: Surface,
                      color: pygame.Color | int | str | tuple[int, int, int] | tuple[int, int, int, int] | Sequence[int],
                      percent: float = 0.1) -> Surface:
    "Return a surface after adding an outline of given color. Percentage is how big the outline is."
    w, h = surface.get_size()
    inside_surf = pygame.transform.scale(surface.copy(),
                                         (round(w*(1 - percent)),
                                          round(h*(1 - percent))))
    surface.fill(color)
    half = percent / 2
    surface.blit(inside_surf, (math.floor(w * half), math.floor(h * half)))
    return surface

def auto_crop_clear(surface: Surface,
                    clear: pygame.Color = pygame.Color(0, 0, 0, 0)) -> Surface:
    "Remove unneccicary pixels from image."
    surface = surface.convert_alpha()
    w, h = surface.get_size()
    surface.lock()
    def find_end(iterfunc: Callable[[int], Generator[pygame.Color, None, None]],
                 rangeobj: range) -> int:
        for x in rangeobj:
            if not all(y == clear for y in iterfunc(x)):
                return x
        return x
    column = lambda x: (surface.get_at((x, y)) for y in range(h))
    row    = lambda y: (surface.get_at((x, y)) for x in range(w))
    leftc  = find_end(column, range(w))
    rightc = find_end(column, range(w-1, -1, -1))
    topc   = find_end(row, range(h))
    floorc = find_end(row, range(h-1, -1, -1))
    surface.unlock()
    dim = pygame.rect.Rect(leftc, topc, rightc-leftc, floorc-topc)
    return surface.subsurface(dim)

def get_tile_color(tile_color: int,
                   greyshift: float = GREYSHIFT) -> tuple[float, float, float]:
    "Return the color a given tile should be."
    if tile_color < 0:
        if tile_color == -6:
            return GREY
        color = tile_colorS[abs(tile_color+1)]
        assert isinstance(color[0], int)
        return lerpColor(color, GREY, greyshift)
    elif tile_color < 5:
        color = tile_colorS[tile_color]
        assert isinstance(color[0], int)
        return color
    raise ValueError('Cannot properly return tile colors greater than five!')

def get_tile_symbol_and_color(tile_color: int,
                              greyshift: float = GREYSHIFT) -> tuple[str, tuple[float, float, float]]:
    "Return the color a given tile should be."
    if tile_color < 0:
        if tile_color == -6:
            return ' ', GREY
        symbol, scolor = TILESYMBOLS[abs(tile_color+1)]
        return symbol, lerpColor(scolor, GREY, greyshift)
    elif tile_color <= 5:
        return TILESYMBOLS[tile_color]
    raise ValueError('Cannot properly return tile colors greater than five!')

def add_symbol_to_tile_surf(surf: Surface,
                            tile_color: int,
                            tilesize: int,
                            greyshift: float = GREYSHIFT,
                            font: str = FONT) -> None:
    "Add tile symbol to tule surface"
    symbol, raw_scolor = get_tile_symbol_and_color(tile_color, greyshift)
    r, g, b = raw_scolor
    scolor = int(r), int(g), int(b)
    pyfont = pygame.font.Font(font, math.floor(math.sqrt(tilesize**2*2))-1)

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
    x = w-sw
    y = h-sh

    surf.blit(symbolsurf, (int(x), int(y)))
##    surf.blit(symbolsurf, (0, 0))

##@lru_cache()
def get_tile_image(tile: Tile,
                   tilesize: int,
                   greyshift: float = GREYSHIFT,
                   outlineSize: float = 0.2,
                   font: str = FONT) -> Surface:
    "Return a surface of a given tile."
    cid = tile.color
    if cid < 5:
        color = get_tile_color(cid, greyshift)
    elif cid >= 5:
        colors = tile_colorS[cid]
        assert isinstance(colors[0], tuple)
        color, outline = colors
        surf = outline_rectangle(make_square_surf(color, tilesize), outline, outlineSize)
        # Add tile symbol
        add_symbol_to_tile_surf(surf, cid, tilesize, greyshift, font)

        return surf
    r, g, b = color
    surf = make_square_surf((int(r), int(g), int(b)), tilesize)
    # Add tile symbol
##    add_symbol_to_tile_surf(surf, cid, tilesize, greyshift, font)

    return surf

def set_alpha(surface: Surface, alpha: int) -> Surface:
    "Return a surface by replacing the alpha chanel of it with given alpha value, preserve color."
    surface = surface.copy().convert_alpha()
    w, h = surface.get_size()
    for y in range(h):
        for x in range(w):
            color = tuple(surface.get_at((x, y)))
            r, g, b = color[:3]
            surface.set_at((x, y), pygame.Color(r, g, b, alpha))
    return surface

##@lru_cache()
def get_tile_container_image(
    wh: tuple[int, int],
    back: pygame.Color | int | str | tuple[int, int, int] | tuple[int, int, int, int] | Sequence[int] | None
) -> Surface:
    "Return a tile container image from a width and a heigth and a background color, and use a game's cache to help."
    image = pygame.surface.Surface(wh)
    image.convert_alpha()
    image = set_alpha(image, 0)

    if not back is None:
        image.convert()
        image.fill(back)
    return image

class Font:
    "Font object, simplify using text."
    def __init__(self,
                 fontName: str,
                 font_size: int = 20,
                 color: pygame.Color | tuple[int, int, int] = (0, 0, 0),
                 cx: bool = True,
                 cy: bool = True,
                 antialias: bool = False,
                 background: pygame.Color | tuple[int, int, int] | None = None,
                 doCache: bool = True):
        self.font = fontName
        self.size = int(font_size)
        self.color = color
        self.center = [cx, cy]
        self.antialias = bool(antialias)
        self.background = background
        self.doCache = bool(doCache)
        self.cache: Surface | None = None
        self.last_text: str | None = None

        self._changeFont()

    def __repr__(self) -> str:
        return 'Font(%r, %i, %r, %r, %r, %r, %r, %r)' % (self.font, self.size, self.color, self.center[0], self.center[1], self.antialias, self.background, self.doCache)

    def _changeFont(self) -> None:
        "Set self.pyfont to a new pygame.font.Font object from data we have."
        self.pyfont = pygame.font.Font(self.font, self.size)

    def _cache(self, surface: Surface) -> None:
        "Set self.cache to surface"

        self.cache = surface

    def get_height(self) -> int:
        "Return the height of font."
        return self.pyfont.get_height()

    def render_nosurf(self,
                      text: str,
                      size: int | None = None,
                      color: pygame.Color | tuple[int, int, int] | None = None,
                      background: pygame.Color | tuple[int, int, int] | None = None,
                      force_update: bool = False) -> Surface:
        "Render and return a surface of given text. Use stored data to render, if arguments change internal data and render."
        updateCache = self.cache is None or force_update or text != self.last_text
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
                self.last_text = text
                surf = self.pyfont.render(text, self.antialias, self.color, self.background).convert_alpha()
                self._cache(surf.copy())
            else:
                assert self.cache is not None
                surf = self.cache
        else:
            # Render the text using the pygame font
            surf = self.pyfont.render(text, self.antialias, self.color, self.background).convert_alpha()
        return surf

    def render(self,
               surface: Surface,
               text: str,
               xy: tuple[int, int],
               size: int | None = None,
               color: pygame.Color | tuple[int, int, int] | None = None,
               background: pygame.Color | tuple[int, int, int] | None = None,
               force_update: bool = False) -> None:
        surf = self.render_nosurf(text, size, color, background, force_update)

        if True in self.center:
            x: int | float = xy[0]
            y: int | float = xy[1]
            cx, cy = self.center
            w, h = surf.get_size()
            if cx:
                x -= w/2
            if cy:
                y -= h/2
            xy = (int(x), int(y))

        surface.blit(surf, xy)


class Object(sprite.Sprite):
    "Object object."
    def __init__(self, name: str) -> None:
        """Sets self.name to name, and other values for rendering.

           Defines the following attributes:
            self.name
            self.image
            self.location
            self.wh
            self.visible
            self.loc_mod_on_resize
            self.id"""
        super().__init__(name)
        self.location = Vector2(round(SCREENSIZE[0]/2), round(SCREENSIZE[1]/2))
        self.loc_mod_on_resize = 'Scale'
        self.screen_size_last = SCREENSIZE

        self.id = 0

        self.add_handler('VideoResize', self.handle_video_resize)

    def __repr__(self) -> str:
        "Return {self.name}()."
        return f'{self.name}()'

    def point_intersects(self, screen_loc: tuple[int, int]) -> bool:
        "Return True if this Object intersects with a given screen location."
        return self.rect.collidepoint(screen_loc)

    def get_screen_location(self, screen_loc: tuple[int, int]) -> tuple[int, int]:
        "Return the location a screen location would be at on the objects image. Can return invalid data."
        # Get zero zero in image locations
        zx, zy = self.location#Zero x and y
        sx, sy = screen_loc#Screen x and y
        return sx - zx, sy - zy#Location with respect to image dimentions

    async def handle_video_resize(self, event: Event[tuple[int, int]]) -> None:
        "Function called when screensize is changed."
        nx, ny = self.location

        if self.loc_mod_on_resize == 'Scale':
            ow, oh = self.screen_size_last
            nw, nh = event['size']

            x, y = self.location
            nx, ny = x * (nw / ow), y * (nh / oh)

        self.location = (nx, ny)
        self.screen_size_last = event['size']


class ObjectHandler(sprite.Group):
    "ObjectHandler class, ment to be used for other classes."
##    __slots__ = ('objects', 'next_id')
    def __init__(self, name: str, *sprites: sprite.Sprite, **kwargs: Any) -> None:
        super().__init__(name, *sprites, **kwargs)

        self.objects: dict[int, str] = {}
        self.next_id = 0
        self.cache: dict[str, int] = {}

    def add_object(self, obj: Object) -> None:
        "Add an object to the game."
        try:
            setattr(obj, 'id', self.next_id)
        except AttributeError:
            pass
        self.objects[self.next_id] = obj.name
        self.next_id += 1
        self.add(obj)

    def rm_object(self, obj: Object) -> None:
        "Remove an object from the game."
        del self.objects[obj.id]
        self.remove(obj)
        self.cache = {}

    def get_object(self, object_id: int) -> Object | None:
        "Return the object accociated with object id given. Return None if object not found."
        if object_id in self.objects:
            component = self.component(self.objects[object_id])
            assert isinstance(component, Object)
            return component
        return None

    def rm_star(self) -> None:
        "Remove all objects from self.objects."
        for oid in list(self.objects):
            obj = self.get_object(oid)
            if obj is None:
                continue
            self.rm_object(obj)
        self.next_id = 0

    def get_objects_with_attr(self, attribute: str) -> tuple[int, ...]:
        "Return a tuple of object ids with given attribute."
        return tuple((oid for oid in self.objects if hasattr(self.get_object(oid), attribute)))

    def get_object_by_attr(self, attribute: str, value: Any) -> tuple[int, ...]:
        "Return a tuple of object ids with <attribute> that are equal to <value>."
        matches = []
        for oid in self.get_objects_with_attr(attribute):
            if getattr(self.get_object(oid), attribute) == value:
                matches.append(oid)
        return tuple(matches)

    def get_object_given_name(self, name: str) -> tuple[int, ...]:
        "Returns a tuple of object ids with names matching <name>."
        return self.get_object_by_attr('name', name)

    def get_object_by_name(self, obj_name: str) -> Object:
        "Get object by name, with cache."
        if not obj_name in self.cache:
            ids = self.get_object_given_name(obj_name)
            if ids:
                self.cache[obj_name] = min(ids)
            else:
                raise RuntimeError(f'{obj_name} Object Not Found!')
        obj = self.get_object(self.cache[obj_name])
        assert obj is not None
        return obj

    def set_attr_all(self, attribute: str, value: Any) -> None:
        "Set given attribute in all of self.objects to given value in all objects with that attribute."
        for oid in self.get_objects_with_attr(attribute):
            setattr(self.get_object(oid), attribute, value)

    def __del__(self) -> None:
        self.rm_star()


class MultipartObject(Object, ObjectHandler):
    "Thing that is both an Object and an ObjectHandler, and is ment to be an Object made up of multiple Objects."
    def __init__(self, name: str, *sprites: sprite.Sprite, **kwargs: Any) -> None:
        """Initalize Object and ObjectHandler of self.

           Also set self._lastloc and self._lastvisible to None"""
##        ObjectHandler.__init__(self, name, *sprites, **kwargs)
        Object.__init__(self, name)

        self._lastloc: tuple[int, int] | None = None
        self._lastvisible: bool = True
        self._to_add: list[Object] = []
        self._rewrite_tick: Callable[[Event], Awaitable[Any | None]] | None = None

    async def temporary_tick_handler(self, event: Event[float]) -> None:
        "Temporary tick handler to add objects before manager set (init block)"
        if self.manager is None:
            return
        for obj in self._to_add:
            self.add_object(obj)
        self._to_add.clear()
        self.remove_handler('tick')
        if self._rewrite_tick is not None:
            self.add_handler('tick', self._rewrite_tick)
            self._rewrite_tick = None

    def add_object(self, obj: Object) -> None:
        "Add an object to the game."
        if self.manager is None:
            self._to_add.append(obj)
            if self._rewrite_tick is not None:
                return
            self._rewrite_tick = self.remove_handler('tick')
            self.add_handler('tick', self.temporary_tick_handler)
            return
        self.manager.add_object(obj)

    @property
    def objects(self) -> dict[int, str]:
        return self.manager.objects

    @property
    def cache(self) -> dict[str, int]:
        return self.manager.cache

    def rm_object(self, obj: Object) -> None:
        "Remove an object from the game."
        self.manager.rm_object(obj)

    def get_object(self, object_id: int) -> Object | None:
        "Return the object accociated with object id given. Return None if object not found."
        return self.manager.get_object(object_id)

    def reset_position(self) -> None:
        "Reset the position of all objects within."
        raise NotImplemented

    def get_where_touches(self, point: tuple[int, int]) -> tuple[str | None, Any | None]:
        "Return where a given point touches in self. Returns (None, None) with no intersections."
        for oid in self.objects:
            obj = self.get_object(oid)
            if obj is None:
                continue
            if hasattr(obj, 'get_tile_point'):
                output = obj.get_tile_point(point)
                if not output is None:
                    return obj.name, output
            else:
                raise Warning('Not all of self.objects have the get_tile_point attribute!')
        return None, None

    def process(self, time_passed: float) -> None:
        "Process Object self and ObjectHandler self and call self.reset_position on location change."

        if self.location != self._lastloc:
            self.reset_position()
            self._lastloc = self.location

        if bool(self.visible) != self._lastvisible:
            self.set_attr_all('visible', self.visible)
            self._lastvisible = self.visible == 1

    def __del__(self) -> None:
        Object.__del__(self)
        ObjectHandler.__del__(self)


class TileRenderer(Object):
    "Base class for all objects that need to render tiles."
    greyshift = GREYSHIFT
    tile_size = TILESIZE
    def __init__(
        self,
        name: str,
        game: "Client",
        tile_seperation: str | float = 'Auto',
        background: tuple[int, int, int] | None = TILEDEFAULT
    ) -> None:
        """Initialize renderer. Needs a game object for its cache and optional tile seperation value and background RGB color.

           Defines the following attributes during initialization and uses throughout:
            self.game
            self.wh
            self.tile_sep
            self.tile_full
            self.back
            and finally, self.imageUpdate

           The following functions are also defined:
            self.clear_image
            self.render_tile
            self.update_image (but not implemented)
            self.process"""
        super().__init__(name)
        self.game = game

        if tile_seperation == 'Auto':
            self.tile_sep = self.tile_size/3.75
        else:
            assert isinstance(tile_seperation, float)
            self.tile_sep = tile_seperation

        self.tile_full = self.tile_size+self.tile_sep
        self.back = background

        self.imageUpdate = True

        self.add_handler('tick', self.handle_tick)

    def clear_image(self, tile_dimentions: tuple[int, int]) -> None:
        "Reset self.image using tile_dimentions tuple and fills with self.back. Also updates self.wh."
        tw, th = tile_dimentions
        self.rect.size = (round(tw*self.tile_full+self.tile_sep), round(th*self.tile_full+self.tile_sep))
        self.image = get_tile_container_image(self.rect.size, self.back)

    def render_tile(self, tile_obj: Tile, tileLoc: tuple[int, int]) -> None:
        "Blit the surface of a given tile object onto self.image at given tile location. It is assumed that all tile locations are xy tuples."
        x, y = tileLoc
        surf = get_tile_image(tile_obj, self.tile_size, self.greyshift)
        self.image.blit(surf, (round(x*self.tile_full+self.tile_sep), round(y*self.tile_full+self.tile_sep)))
        self.dirty = 1

    def update_image(self) -> None:
        "Called when processing image changes, directed by self.imageUpdate being True."
        raise NotImplemented

    async def handle_tick(self, event: Event[float]) -> None:
        "Call self.update_image() if self.imageUpdate is True, then set self.update_image to False."
        if self.imageUpdate:
            self.update_image()
            self.imageUpdate = False


class Cursor(TileRenderer):
    "Cursor Object."
    greyshift = GREYSHIFT
    def __init__(self, game: "Client") -> None:
        "Initialize cursor with a game it belongs to."
        TileRenderer.__init__(self, 'Cursor', game, 'Auto', None)

        self.holding_number_one = False
        self.tiles: deque[Tile] = deque()
        self.visible = 0

    def update_image(self) -> None:
        "Update self.image."
        self.clear_image((len(self.tiles), 1))

        for x in range(len(self.tiles)):
            self.render_tile(self.tiles[x], (x, 0))

    def is_pressed(self) -> bool:
        "Return True if the right mouse button is pressed."
        return bool(pygame.mouse.get_pressed()[0])

    def get_held_count(self, count_one: bool = False) -> int:
        "Return the number of held tiles, can be discounting number one tile."
        l = len(self.tiles)
        if self.holding_number_one and not count_one:
            return l-1
        return l

    def is_holding(self, count_one: bool = False) -> bool:
        "Return True if the mouse is dragging something."
        return self.get_held_count(count_one) > 0

    def get_held_info(self, count_one: bool = False) -> tuple[Tile | None, int]:
        "Returns color of tiles are and number of tiles held."
        if not self.is_holding(count_one):
            return None, 0
        return self.tiles[0], self.get_held_count(count_one)

    async def handle_tick(self, event: Event[float]) -> None:
        "Process cursor."
        self.location = pygame.mouse.get_pos()
        self.visible = int(self.is_holding(True))
        await super().handle_tick(event)

    def force_hold(self, tiles: Sequence[Tile]) -> None:
        "Pretty much it's drag but with no constraints."
        for tile in tiles:
            if tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.imageUpdate = True

    def drag(self, tiles: Sequence[Tile]) -> None:
        "Drag one or more tiles, as long as it's a list."
        for tile in tiles:
            if not tile is None and tile.color == NUMBERONETILE:
                self.holding_number_one = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.imageUpdate = True

    def drop(self, number: str | int = 'All', count_one: bool = False) -> list[Tile]:
        "Return all of the tiles the Cursor is carrying"
        if self.is_holding(count_one):
            if number == 'All':
                number = self.get_held_count(count_one)
            else:
                assert isinstance(number, int)
                number = saturate(number, 0, self.get_held_count(count_one))

            tiles = []
            for tile in (self.tiles.popleft() for i in range(number)):
                if tile.color == NUMBERONETILE:
                    if not count_one:
                        self.tiles.append(tile)
                        continue
                tiles.append(tile)
            self.imageUpdate = True

            self.holding_number_one = NUMBERONETILE in {tile.color for tile in self.tiles}
            return tiles
        return []

    def drop_one_tile(self) -> Tile | None:
        "If holding the number one tile, drop it (returns it)."
        if self.holding_number_one:
            count_one = self.drop('All', False)
            one = self.drop(1, True)
            self.drag(count_one)
            self.holding_number_one = False
            return one[0]
        return None


def gscBoundIndex(
    bounds_fail_value: object = None
) -> Callable[[Callable[[Grid, tuple[int, int]], Any]], Callable[[Grid, tuple[int, int]], Any]]:
    "Return a decorator for any grid or grid subclass that will keep index positions within bounds."
    def gsc_keep_bounds(function: Callable[["Grid", tuple[int, int]], Any]) -> Callable[[Grid, tuple[int, int]], Any]:
        "Grid or Grid Subclass Decorator that keeps index positions within bounds, as long as index is first argument after self arg."
        @wraps(function)
        def keep_bounds(self: "Grid", index: tuple[int, int], *args: Any, **kwargs: Any) -> Any:
            "Wraper function that makes sure a position tuple (x, y) is valid."
            x, y = index
            if x < 0 or x >= self.size[0]:
                return bounds_fail_value
            if y < 0 or y >= self.size[1]:
                return bounds_fail_value
            return function(self, index, *args, **kwargs)
        return keep_bounds
    @wraps(gsc_keep_bounds)
    def get_gsc_bounds_wrapper(function: Callable[["Grid", tuple[int, int]], Any]) -> Callable[["Grid", tuple[int, int]], Any]:
        return gsc_keep_bounds(function)
    return get_gsc_bounds_wrapper

class Grid(TileRenderer):
    "Grid object, used for boards and parts of other objects."
    def __init__(
        self,
        name: str,
        size: tuple[int, int],
        game: "Client",
        tile_seperation: str | float = 'Auto',
        background: tuple[int, int, int] | None = TILEDEFAULT
    ) -> None:
        "Grid Objects require a size and game at least."
        TileRenderer.__init__(self, name, game, tile_seperation, background)

        self.size = size

        self.data = array([Tile(-6) for i in range(int(self.size[0]*self.size[1]))]).reshape(self.size)

    def update_image(self) -> None:
        "Update self.image."
        self.clear_image(self.size)

        for y in range(self.size[1]):
            for x in range(self.size[0]):
                self.render_tile(self.data[x, y], (x, y))

    def get_tile_point(self, screen_loc: tuple[int, int]) -> tuple[int, int] | None:
        "Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.point_intersects(screen_loc):
            return None
        # Otherwise, find out where screen point is in image locations
        bx, by = self.get_screen_location(screen_loc)#board x and y
        # Finally, return the full divides (no decimals) of xy location by self.tile_full.
        return int(bx // self.tile_full), int(by // self.tile_full)

    @gscBoundIndex()
    def place_tile(self, xy: tuple[int, int], tile: Tile) -> bool:
        "Place a Tile Object if permitted to do so. Return True if success."
        x, y = xy
        if self.data[x, y].color < 0:
            self.data[x, y] = tile
            del tile
            self.imageUpdate = True
            return True
        return False

    @gscBoundIndex()
    def get_tile(self, xy: tuple[int, int], replace: int = -6) -> Tile | None:
        "Return a Tile Object from a given position in the grid if permitted. Return None on falure."
        x, y = xy
        tile_copy: Tile = self.data[x, y]
        if tile_copy.color < 0:
            return None
        self.data[x, y] = Tile(replace)
        self.imageUpdate = True
        return tile_copy

    @gscBoundIndex()
    def get_info(self, xy: tuple[int, int]) -> Tile:
        "Return the Tile Object at a given position without deleteing it from the Grid."
        x, y = xy
        return self.data[x, y]

    def get_colors(self) -> list[int]:
        "Return a list of the colors of tiles within self."
        colors = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                infoTile = self.get_info((x, y))
                if not infoTile.color in colors:
                    colors.append(infoTile.color)
        return colors

    def is_empty(self, empty_color: int = -6) -> bool:
        "Return True if Grid is empty (all tiles are empty_color)."
        colors = self.get_colors()
        # Colors should only be [-6] if empty
        return colors == [empty_color]

    def __del__(self) -> None:
        super().__del__()
        del self.data


class Board(Grid):
    "Represents the board in the Client."
    size = (5, 5)
    bcolor = ORANGE
    def __init__(self, player: "Player", variant_play: bool = False) -> None:
        "Requires a player object."
        Grid.__init__(self, 'Board', self.size, player.game, background=self.bcolor)
        self.player = player

        self.variant_play = variant_play
        self.additions: dict[int, Tile] = {}

        self.wall_tileing = False

    def __repr__(self) -> str:
        return f'Board({self.player}, {self.variant_play})'

    def set_colors(self, real: bool = True) -> None:
        "Reset tile colors."
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if not real or self.data[x, y].color < 0:
                    self.data[x, y].color = -((self.size[1]-y+x)%REGTILECOUNT+1)
##                print(self.data[x, y].color, end=' ')
##            print()
##        print('-'*10)

    def get_row(self, index: int) -> list[Tile]:
        "Return a row from self. Does not delete data from internal grid."
        return [self.get_info((x, index)) for x in range(self.size[0])]

    def get_col(self, index: int) -> list[Tile]:
        "Return a column from self. Does not delete data from internal grid."
        return [self.get_info((index, y)) for y in range(self.size[1])]

    def get_colors_in_row(self, index: int, no_negative: bool = True) -> list[int]:
        "Return the colors placed in a given row in internal grid."
        row_colors = [tile.color for tile in self.get_row(index)]
        if no_negative:
            row_colors = [c for c in row_colors if c >= 0]
        ccolors = Counter(row_colors)
        return list(sorted(ccolors.keys()))

    def get_colors_column(self, index: int, no_negative: bool = True) -> list[int]:
        "Return the colors placed in a given row in internal grid."
        col_colors = [tile.color for tile in self.get_col(index)]
        if no_negative:
            col_colors = [c for c in col_colors if c >= 0]
        ccolors = Counter(col_colors)
        return list(sorted(ccolors.keys()))

    def is_wall_tileing(self) -> bool:
        "Return True if in Wall Tiling Mode."
        return self.wall_tileing

    def get_tile_for_cursor_by_row(self, row: int) -> Tile | None:
        "Return A COPY OF tile the mouse should hold. Returns None on failure."
        if row in self.additions:
            data = self.additions[row]
            if isinstance(data, Tile):
                return data
        return None

    @gscBoundIndex(False)
    def is_color_point_valid(self, position: tuple[int, int], tile: Tile) -> bool:
        "Return True if tile's color is valid at given position."
        column, row = position
        colors = set(self.get_colors_column(column) + self.get_colors_in_row(row))
        return not tile.color in colors

    def get_tile_rows(self) -> dict[int, int]:
        "Return a dictionary of row numbers and row color to be wall tiled."
        rows = {}
        for row in self.additions:
            if isinstance(self.additions[row], Tile):
                rows[row] = self.additions[row].color
        return rows

    def get_valid_tile_row_locs(self, row: int) -> tuple[int, ...]:
        "Return the valid drop columns of the additions tile for a given row."
        valid = []
        if row in self.additions:
            tile = self.additions[row]
            if isinstance(tile, Tile):
                for column in range(self.size[0]):
                    if self.is_color_point_valid((column, row), tile):
                        valid.append(column)
                return tuple(valid)
        return ()

    def remove_unplaceable_additions(self) -> None:
        "Remove invalid additions that would not be placeable."
        # In the wall-tiling phase, it may happen that you
        # are not able to move the rightmost tile of a certain
        # pattern line over to the wall because there is no valid
        # space left for it. In this case, you must immediately
        # place all tiles of that pattern line in your floor line.
        for row in range(self.size[1]):
            if isinstance(self.additions[row], Tile):
                valid = self.get_valid_tile_row_locs(row)
                if not valid:
                    floor = self.player.get_object_by_name('FloorLine')
                    floor.place_tile(self.additions[row])
                    self.additions[row] = None

    @gscBoundIndex(False)
    def wall_tile_from_point(self, position: tuple[int, int]) -> bool:
        "Given a position, wall tile. Return success on placement. Also updates if in wall tiling mode."
        success = False
        column, row = position
        atPoint = self.get_info(position)
        if atPoint.color <= 0:
            if row in self.additions:
                tile = self.additions[row]
                if not tile is None:
                    if self.is_color_point_valid(position, tile):
                        self.place_tile(position, tile)
                        self.additions[row] = column
                        # Update invalid placements after new placement
                        self.remove_unplaceable_additions()
                        success = True
        if not self.get_tile_rows():
            self.wall_tileing = False
        return success

    def wall_tile_mode(self, moved: Any) -> None:
        "Set self into Wall Tiling Mode. Finishes automatically if not in varient play mode."
        self.wall_tileing = True
        for key, value in ((key, moved[key]) for key in moved.keys()):
            key = int(key)-1
            if key in self.additions:
                raise RuntimeError('Key %r Already in additions dictionary!' % key)
            self.additions[key] = value
        if not self.variant_play:
            for row in range(self.size[1]):
                if row in self.additions:
                    rowdata = [tile.color for tile in self.get_row(row)]
                    tile = self.additions[row]
                    if tile is None:
                        continue
                    negtile_color = -(tile.color+1)
                    if negtile_color in rowdata:
                        column = rowdata.index(negtile_color)
                        self.place_tile((column, row), tile)
                        # Set data to the column placed in, use for scoring
                        self.additions[row] = column
                    else:
                        raise RuntimeError('%i not in row %i!' % (negtile_color, row))
                else:
                    raise RuntimeError(f'{row} not in moved!')
            self.wall_tileing = False
        else:
            # Invalid additions can only happen in variant play mode.
            self.remove_unplaceable_additions()


    @gscBoundIndex(([], []))
    def getTouchesContinuous(self, xy: tuple[int, int]) -> tuple[list[int], list[int]]:
        "Return two lists, each of which contain all the tiles that touch the tile at given x y position, including that position."
        rs, cs = self.size
        x, y = xy
        # Get row and column tile color data
        row = [tile.color for tile in self.get_row(y)]
        column = [tile.color for tile in self.get_col(x)]
        # Both
        def gt(v, size, data):
            "Go through data foreward and backward from point v out by size, and return all points from data with a value >= 0."
            def trng(rng, data):
                "Try range. Return all of data in range up to when indexed value is < 0."
                ret = []
                for tv in rng:
                    if data[tv] < 0:
                        break
                    ret.append(tv)
                return ret
            nt = trng(reversed(range(0, v)), data)
            pt = trng(range(v+1, size), data)
            return nt + pt
        # Combine two lists by zipping together and returning list object.
        comb = lambda one, two: list(zip(one, two))
        # Return all of the self.get_info points for each value in lst.
        getAll = lambda lst: [self.get_info(pos) for pos in lst]
        # Get row touches
        rowTouches = comb(gt(x, rs, row), [y]*rs)
        # Get column touches
        columnTouches = comb([x]*cs, gt(y, cs, column))
        # Get real tiles from indexes and return
        return getAll(rowTouches), getAll(columnTouches)

    def scoreAdditions(self):
        "Using self.additions, which is set in self.wall_tile_mode(), return the number of points the additions scored."
        score = 0
        for x, y in ((self.additions[y], y) for y in range(self.size[1])):
            if not x is None:
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

    def get_filled_rows(self):
        "Return the number of filled rows on this board."
        count = 0
        for row in range(self.size[1]):
            real = (t.color >= 0 for t in self.get_row(row))
            if all(real):
                count += 1
        return count

    def hasFilledRow(self):
        "Return True if there is at least one completely filled horizontal line."
        return self.get_filled_rows() >= 1

    def get_filled_cols(self):
        "Return the number of filled rows on this board."
        count = 0
        for column in range(self.size[0]):
            real = (t.color >= 0 for t in self.get_col(column))
            if all(real):
                count += 1
        return count

    def get_filled_colors(self):
        "Return the number of completed colors on this board."
        tiles = (self.get_info((x, y)) for x in range(self.size[0]) for y in range(self.size[1]))
        colors = [t.color for t in tiles]
        colorCount = Counter(colors)
        count = 0
        for fillNum in colorCount.values():
            if fillNum >= 5:
                count += 1
        return count

    def end_score(self):
        "Return the additional points for this board at the end of the game."
        score = 0
        score += self.get_filled_rows() * 2
        score += self.get_filled_cols() * 7
        score += self.get_filled_colors() * 10
        return score

    def process(self, time_passed):
        "Process board."
        if self.imageUpdate and not self.variant_play:
            self.set_colors(True)
        super().process(time_passed)

    def get_data(self):
        "Return data that makes this Grid Object special. Compress tile data by getting color values plus seven, then getting the hex of that as a string."
        data = super().get_data()
        data['Wt'] = int(self.wall_tileing)
        adds = ''
        for t in self.additions.values():
            if t is None:#If none, n
                adds += 'n'
            elif isinstance(t, Tile):# If tile, a to l
                adds += chr(t.color+6+65)#97)
            elif isinstance(t, int):# If int, return string repr of value.
                if t > 9:
                    raise RuntimeError(f'Intiger overflow with value {t} > 9!')
                adds += str(t)
            else:
                raise RuntimeError(f'Invalid additions value "{t}"!')
        data['Ad'] = adds
        return data

    def from_data(self, data):
        "Update this Board object from data."
        super().from_data(data)
        self.wall_tileing = bool(data['Wt'])
        for k in range(len(data['Ad'])):
            rv = data['Ad'][k]
            if rv == 'n':
                v = None
            elif rv.isupper():
                v = Tile(ord(rv)-65-6)
            else:
                v = int(rv)
            self.additions[k] = v


class Row(TileRenderer):
    "Represents one of the five rows each player has."
    greyshift = GREYSHIFT
    def __init__(self, name: str, player, size, tilesep='Auto', background=None):
        TileRenderer.__init__(self, name, player.game, tilesep, background)
        self.player = player
        self.size = int(size)

        self.color = -6
        self.tiles = deque([Tile(self.color)]*self.size)

    def __repr__(self):
        return 'Row(%r, %i, ...)' % (self.game, self.size)

    @classmethod
    def from_list(cls, player, iterable):
        "Return a new Row Object from a given player and an iterable of tiles."
        lst = deque(iterable)
        obj = cls(player, len(lst))
        obj.color = None
        obj.tiles = lst
        return obj

    def update_image(self) -> None:
        "Update self.image."
        self.clear_image((self.size, 1))

        for x in range(len(self.tiles)):
            self.render_tile(self.tiles[x], (x, 0))

    def get_tile_point(self, screen_loc):
        "Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."
        xy = Grid.get_tile_point(self, screen_loc)
        if xy is None:
            return None
        x, y = xy
        return self.size-1-x

    def get_placed(self):
        "Return the number of tiles in self that are not fake tiles, like grey ones."
        return len([tile for tile in self.tiles if tile.color >= 0])

    def get_placeable(self):
        "Return the number of tiles permitted to be placed on self."
        return self.size - self.get_placed()

    def is_full(self):
        "Return True if this row is full."
        return self.get_placed() == self.size

    def get_info(self, location):
        "Return tile at location without deleteing it. Return None on invalid location."
        index = self.size-1-location
        if index < 0 or index > len(self.tiles):
            return None
        return self.tiles[index]

    def can_place(self, tile):
        "Return True if permitted to place given tile object on self."
        placeable = (tile.color == self.color) or (self.color < 0 and tile.color >= 0)
        colorCorrect = tile.color >= 0 and tile.color < 5
        numCorrect = self.get_placeable() > 0

        board = self.player.get_object_by_name('Board')
        colorNotPresent = not tile.color in board.get_colors_in_row(self.size-1)

        return placeable and colorCorrect and numCorrect and colorNotPresent

    def get_tile(self, replace=-6):
        "Return the leftmost tile while deleteing it from self."
        self.tiles.appendleft(Tile(replace))
        self.imageUpdate = True
        return self.tiles.pop()

    def place_tile(self, tile):
        "Place a given Tile Object on self if permitted."
        if self.can_place(tile):
            self.color = tile.color
            self.tiles.append(tile)
            end = self.tiles.popleft()
            if not end.color < 0:
                raise RuntimeError('Attempted deleteion of real tile from Row!')
            self.imageUpdate = True

    def can_place_tiles(self, tiles):
        "Return True if permitted to place all of given tiles objects on self."
        if len(tiles) > self.get_placeable():
            return False
        for tile in tiles:
            if not self.can_place(tile):
                return False
        tile_colors = []
        for tile in tiles:
            if not tile.color in tile_colors:
                tile_colors.append(tile.color)
        if len(tile_colors) > 1:
            return False
        return True

    def place_tiles(self, tiles):
        "Place multiple tile objects on self if permitted."
        if self.can_place_tiles(tiles):
            for tile in tiles:
                self.place_tile(tile)

    def wallTile(self, addToDict, blankColor=-6):
        "Move tiles around and into add dictionary for the wall tiling phase of the game. Removes tiles from self."
        if not 'toBox' in addToDict:
            addToDict['toBox'] = []
        if not self.is_full():
            addToDict[str(self.size)] = None
            return
        else:
            self.color = blankColor
        addToDict[str(self.size)] = self.get_tile()
        for i in range(self.size-1):
            addToDict['toBox'].append(self.get_tile())

    def set_background(self, color):
        "Set the background color for this row."
        self.back = color
        self.imageUpdate = True


class PatternLine(MultipartObject):
    "Represents multiple rows to make the pattern line."
    size = (5, 5)
    def __init__(self, player, row_sep=0):
        MultipartObject.__init__(self, 'PatternLine')
        self.player = player
        self.rowSep = row_sep

        for x, y in zip(range(self.size[0]), range(self.size[1])):
            self.add_object(Row(f'Row{x}_{y}', self.player, x+1))

##        self.set_background(None)

        self._lastloc = 0, 0

    def set_background(self, color):
        "Set the background color for all rows in the pattern line."
        self.set_attr_all('back', color)
        self.set_attr_all('imageUpdate', True)

    def get_row(self, row):
        "Return given row."
        return self.get_object(row)

    def reset_position(self):
        "Reset Locations of Rows according to self.location."
        last = self.size[1]
        w = self.get_row(last-1).wh[0]
        if w is None:
            raise RuntimeError('Image Dimentions for Row Object (row.wh) are None!')
        h1 = self.get_row(0).tile_full
        h = last*h1
        self.wh = w, h
        w1 = h1/2

        x, y = self.location
        y -= h/2-w1
        for rid in self.objects:
            l = last-self.objects[rid].size
            self.objects[rid].location = x+(l*w1), y+rid*h1

    def get_tile_point(self, screen_loc):
        "Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."
        for y in range(self.size[1]):
            x = self.get_row(y).get_tile_point(screen_loc)
            if not x is None:
                return x, y

    def is_full(self):
        "Return True if self is full."
        for rid in range(self.size[1]):
            if not self.get_row(rid).is_full():
                return False
        return True

    def wall_tileing(self):
        "Return a dictionary to be used with wall tiling. Removes tiles from rows."
        values = {}
        for rid in range(self.size[1]):
            self.get_row(rid).wallTile(values)
        return values

    def process(self, time_passed):
        "Process all the rows that make up the pattern line."
        if self.visible != self._lastvisible:
            self.set_attr_all('imageUpdate', True)
        super().process(time_passed)


class Text(Object):
    "Text object, used to render text with a given font."
    def __init__(self, name: str, font_size, color, background=None, cx=True, cy=True):
        super().__init__(name)
        self.font = Font(FONT, font_size, color, cx, cy, True, background, True)
        self._cxy = cx, cy
        self._last = None

    @staticmethod
    def get_font_height(font, size: int) -> int:
        "Return the height of font at font_size size."
        return pygame.font.Font(font, size).get_height()

    def update_value(self, text: str, size=None, color=None, background='set') -> None:
        "Return a surface of given text rendered in FONT."
        if background == 'set':
            self.image = self.font.render_nosurf(text, size, color)
            return
        self.image = self.font.render_nosurf(text, size, color, background)

    def get_tile_point(self, location):
        "Set get_tile_point attribute so that errors are not raised."
        return None


class FloorLine(Row):
    "Represents a player's floor line."
    size = 7
    number_oneColor = NUMBERONETILE
    def __init__(self, player):
        Row.__init__(self, 'FloorLine', player, self.size, background=ORANGE)

##        self.font = Font(FONT, round(self.tile_size*1.2), color=BLACK, cx=False, cy=False)
        self.text = Text("Floor Text", round(self.tile_size*1.2), BLACK, cx=False, cy=False)
        self.hasNumberOne = False

        gen = floorLineSubGen(1)
        self.numbers = [next(gen) for i in range(self.size)]

    def __repr__(self):
        return 'FloorLine(%r)' % self.player

    def render(self, surface):
        "Update self.image."
        Row.render(self, surface)

        sx, sy = self.location
        if self.wh is None:
            return
        w, h = self.wh
        for x in range(self.size):
            xy = round(x*self.tile_full+self.tile_sep+sx-w/2), round(self.tile_sep+sy-h/2)
            self.text.update_value(str(self.numbers[x]))
            self.text.location = xy
            self.text.render(surface)
##            self.font.render(surface, str(self.numbers[x]), xy)

    def place_tile(self, tile):
        "Place a given Tile Object on self if permitted."
        self.tiles.insert(self.get_placed(), tile)

        if tile.color == self.number_oneColor:
            self.hasNumberOne = True

        boxLid = self.player.game.get_object_by_name('BoxLid')

        def handleEnd(end):
            "Handle the end tile we are replacing. Ensures number one tile is not removed."
            if not end.color < 0:
                if end.color == self.number_oneColor:
                    handleEnd(self.tiles.pop())
                    self.tiles.appendleft(end)
                    return
                boxLid.add_tile(end)

        handleEnd(self.tiles.pop())

        self.imageUpdate = True

    def scoreTiles(self):
        "Score self.tiles and return how to change points."
        runningTotal = 0
        for x in range(self.size):
            if self.tiles[x].color >= 0:
                runningTotal += self.numbers[x]
            elif x < self.size-1:
                if self.tiles[x+1].color >= 0:
                    raise RuntimeError('Player is likely cheating! Invalid placement of FloorLine tiles!')
        return runningTotal

    def get_tiles(self, emtpyColor=-6):
        "Return tuple of tiles gathered, and then either the number one tile or None."
        tiles = []
        number_oneTile = None
        for tile in (self.tiles.pop() for i in range(len(self.tiles))):
            if tile.color == self.number_oneColor:
                number_oneTile = tile
                self.hasNumberOne = False
            elif tile.color >= 0:
                tiles.append(tile)

        for i in range(self.size):
            self.tiles.append(Tile(emtpyColor))
        self.imageUpdate = True
        return tiles, number_oneTile

    def can_place_tiles(self, tiles):
        "Return True."
        return True

    def get_data(self):
        "Return the data that makes this FloorLine Row special."
        data = super().get_data()
        data['fnt'] = self.font.get_data()
        return data

    def from_data(self, data):
        "Updata this FloorLine from data."
        super().from_data(data)
        self.font.from_data(data['fnt'])


class Factory(Grid):
    "Represents a Factory."
    size = (2, 2)
    color = WHITE
    outline = BLUE
    outSize = 0.1
    def __init__(self, game, factoryId):
        Grid.__init__(self, f'Factory{self.number}', self.size, game, background=None)
        self.number = factoryId

        self.radius = math.ceil(self.tile_full * self.size[0] * self.size[1] / 3 + 3)

    def __repr__(self):
        return 'Factory(%r, %i)' % (self.game, self.number)

    def addCircle(self, surface):
        if not f'FactoryCircle{self.radius}' in self.game.cache:
            rad = math.ceil(self.radius)
            surf = set_alpha(pygame.surface.Surface((2*rad, 2*rad)), 1)
            pygame.draw.circle(surf, self.outline, (rad, rad), rad)
            pygame.draw.circle(surf, self.color, (rad, rad), math.ceil(rad*(1-self.outSize)))
            self.game.cache[f'FactoryCircle{self.radius}'] = surf
        surf = self.game.cache[f'FactoryCircle{self.radius}'].copy()
        surface.blit(surf, (round(self.location[0]-self.radius), round(self.location[1]-self.radius)))

    def render(self, surface):
        "Render Factory."
        if not self.visible:
            self.addCircle(surface)
        super().render(surface)

    def fill(self, tiles):
        "Fill self with tiles. Will raise exception if insufficiant tiles."
        if len(tiles) < self.size[0] * self.size[1]:
            raise RuntimeError('Insufficiant quantity of tiles! Needs %i!' % self.size[0] * self.size[1])
        for y in range(self.size[1]):
            for tile, x in zip((tiles.pop() for i in range(self.size[0])), range(self.size[0])):
                self.place_tile((x, y), tile)
        if tiles:
            raise RuntimeError('Too many tiles!')

    def grab(self):
        "Return all tiles on this factory."
        return [tile for tile in (self.get_tile((x, y)) for x in range(self.size[0]) for y in range(self.size[1])) if tile.color != -6]

    def grabColor(self, color):
        "Return all tiles of color given in the first list, and all non-matches in the seccond list."
        tiles = self.grab()
        right, wrong = [], []
        for tile in tiles:
            if tile.color == color:
                right.append(tile)
            else:
                wrong.append(tile)
        return right, wrong

    def process(self, time_passed):
        "Process self."
        if self.imageUpdate:
            self.radius = self.tile_full * self.size[0] * self.size[1] / 3 + 3
        super().process(time_passed)

    def get_data(self):
        "Return what makes this Factory Grid special."
        data = super().get_data()
        data['n'] = self.number
        data['r'] = f'{math.ceil(self.radius):x}'
        return data

    def from_data(self, data):
        "Update this Factory from data."
        super().from_data(data)
        self.number = int(data['n'])
        self.name = f'Factory{self.number}'
        self.radius = int(f"0x{data['r']}", 16)


class Factories(MultipartObject):
    "Factories Multipart Object, made of multiple Factory Objects."
    tiles_each = 4
    def __init__(self, game, factories: int, size='Auto') -> None:
        "Requires a number of factories."
        super().__init__('Factories')

        self.game = game
        self.count = factories

        for i in range(self.count):
            self.add_object(Factory(self.game, i))

        if size == 'Auto':
            self.objects[0].process(0)
            rad = self.objects[0].radius
            self.size = rad * 5
        else:
            self.size = size
        self.size = math.ceil(self.size)

        self.divy_up_tiles()

    def __repr__(self) -> str:
        return 'Factories(%r, %i, ...)' % (self.game, self.count)

    def reset_position(self):
        "Reset the position of all factories within."
        degrees = 360 / self.count
        for i in range(self.count):
            rot = math.radians(degrees * i)
            self.objects[i].location = math.sin(rot)*self.size + self.location[0], math.cos(rot)*self.size + self.location[1]

    def process(self, time_passed):
        "Process factories. Does not react to cursor if visible."
        super().process(time_passed)
        if not self.visible:
            cursor = self.game.get_object_by_name('Cursor')
            if cursor.is_pressed() and not cursor.is_holding():
                obj, point = self.get_where_touches(cursor.location)
                if not obj is None and not point is None:
                    oid = int(obj[7:])
                    tileAtPoint = self.objects[oid].get_info(point)
                    if (not tileAtPoint is None) and tileAtPoint.color >= 0:
                        table = self.game.get_object_by_name('TableCenter')
                        select, tocenter = self.objects[oid].grabColor(tileAtPoint.color)
                        if tocenter:
                            table.add_tiles(tocenter)
                        cursor.drag(select)

    def divy_up_tiles(self, empty_color=-6):
        "Divy up tiles to each factory from the bag."
        # For every factory we have,
        for fid in range(self.count):
            # Draw tiles for the factory
            drawn = []
            for i in range(self.tiles_each):
                # If the bag is not empty,
                if not self.game.bag.is_empty():
                    # Draw a tile from the bag.
                    drawn.append(self.game.bag.draw_tile())
                else:# Otherwise, get the box lid
                    boxLid = self.game.get_object_by_name('BoxLid')
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
        "Return True if all factories are empty."
        for fid in range(self.count):
            if not self.objects[fid].is_empty():
                return False
        return True

    def get_data(self):
        "Return what makes this Factories ObjectHandler special."
        data = super().get_data()
        data['cnt'] = f'{self.count:x}'
        data['sz'] = f'{math.ceil(self.size):x}'
        return data

    def from_data(self, data):
        "Update these Factories with data."
        super().from_data(data)
        self.count = int(f"0x{data['cnt']}", 16)
        self.size = int(f"0x{data['sz']}", 16)


class TableCenter(Grid):
    "Object that represents the center of the table."
    size = (6, 6)
    firsttile_color = NUMBERONETILE
    def __init__(self, game, hasOne=True):
        "Requires a game object handler to exist in."
        Grid.__init__(self, 'TableCenter', self.size, game, background=None)
        self.game = game

        self.firstTileExists = False
        if hasOne:
            self.add_number_one_tile()

        self.next_position = (0, 0)

    def __repr__(self):
        return 'TableCenter(%r)' % self.game

    def add_number_one_tile(self):
        "Add the number one tile to the internal grid."
        if not self.firstTileExists:
            x, y = self.size
            self.place_tile((x-1, y-1), Tile(self.firsttile_color))
            self.firstTileExists = True

    def add_tile(self, tile: Tile) -> None:
        "Add a Tile Object to the Table Center Grid."
        self.place_tile(self.next_position, tile)
        x, y = self.next_position
        x += 1
        y += int(x // self.size[0])
        x %= self.size[0]
        y %= self.size[1]
        self.next_position = (x, y)
        self.imageUpdate = True

    def add_tiles(self, tiles, sort=True):
        "Add multiple Tile Objects to the Table Center Grid."
        yes = []
        for tile in tiles:
            self.add_tile(tile)
        if sort and tiles:
            self.reorder_tiles()

    def reorder_tiles(self, replace=-6):
        "Re-organize tiles by Color."
        full = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if self.firstTileExists:
                    if self.get_info((x, y)).color == self.firsttile_color:
                        continue
                at = self.get_tile((x, y), replace)

                if not at is None:
                    full.append(at)
        sortedTiles = sorted(full, key=sortTiles)
        self.next_position = (0, 0)
        self.add_tiles(sortedTiles, False)

    def pull_tiles(self, tile_color, replace=-6):
        "Remove all of the tiles of tile_color from the Table Center Grid."
        toPull = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                infoTile = self.get_info((x, y))
                if infoTile.color == tile_color:
                    toPull.append((x, y))
                elif self.firstTileExists:
                    if infoTile.color == self.firsttile_color:
                        toPull.append((x, y))
                        self.firstTileExists = False
        tiles = [self.get_tile(pos, replace) for pos in toPull]
        self.reorder_tiles(replace)
        return tiles

    def process(self, time_passed):
        "Process factories."
        if not self.visible:
            cursor = self.game.get_object_by_name('Cursor')
            if cursor.is_pressed() and not cursor.is_holding() and not self.is_empty():
                if self.point_intersects(cursor.location):
                    point = self.get_tile_point(cursor.location)
                    # Shouldn't return none anymore since we have point_intersects now.
                    colorAtPoint = self.get_info(point).color
                    if colorAtPoint >= 0 and colorAtPoint < 5:
                        cursor.drag(self.pull_tiles(colorAtPoint))
        super().process(time_passed)

    def get_data(self):
        "Return what makes the TableCenter special."
        data = super().get_data()
        data['fte'] = int(self.firstTileExists)
        x, y = self.next_position
        data['np'] = f'{x}{y}'
        return data

    def from_data(self, data):
        "Update the TableCenter from data."
        super().from_data(data)
        self.firstTileExists = bool(data['fte'])
        x, y = data['np']
        self.next_position = int(x), int(y)


class Bag:
    "Represents the bag full of tiles."
    def __init__(self, num_tiles: int = 100, tile_types: int = 5) -> None:
        self.num_tiles = int(num_tiles)
        self.tile_types = int(tile_types)
        self.tile_names = [chr(65+i) for i in range(self.tile_types)]
        self.percentEach = (self.num_tiles/self.tile_types)/100
        self.full_reset()

    def full_reset(self) -> None:
        "Reset the bag to a full, re-randomized bag."
        self.tiles = gen_random_proper_seq(self.num_tiles, **{tile_name:self.percentEach for tile_name in self.tile_names})

    def __repr__(self) -> str:
        return f'Bag({self.num_tiles}, {self.tile_types})'

    def reset(self) -> None:
        "Randomize all the tiles in the bag."
        self.tiles = deque(randomize(self.tiles))

    def get_color(self, tile_name: str) -> int:
        "Return the color of a named tile."
        if not tile_name in self.tile_names:
            raise ValueError('Tile Name %s Not Found!' % tile_name)
        return self.tile_names.index(tile_name)

    def get_tile(self, tile_name: str) -> Tile:
        "Return a Tile Object from a tile name."
        return Tile(self.get_color(tile_name))

    def get_count(self) -> int:
        "Return number of tiles currently held."
        return len(self.tiles)

    def is_empty(self) -> bool:
        "Return True if no tiles are currently held."
        return self.get_count() == 0

    def draw_tile(self) -> Tile | None:
        "Return a random Tile Object from the bag. Return None if no tiles to draw."
        if not self.is_empty():
            return self.get_tile(self.tiles.pop())
        return None

    def get_name(self, tile_color: int) -> str:
        "Return the name of a tile given it's color."
        try:
            return self.tile_names[tile_color]
        except IndexError:
            raise ValueError('Invalid Tile Color!')

    def add_tile(self, tile_object: Tile) -> None:
        "Add a Tile Object to the bag."
        name = self.get_name(int(tile_object.color))
        rnge = (0, len(self.tiles)-1)
        if rnge[1]-rnge[0] <= 1:
            index = 0
        else:
            index = random.randint(rnge[0], rnge[1])
##        self.tiles.insert(random.randint(0, len(self.tiles)-1), self.get_name(int(tile_object.color)))
        self.tiles.insert(index, name)
        del tile_object

    def add_tiles(self, tile_objects: Sequence[Tile]) -> None:
        "Add multiple Tile Objects to the bag."
        for tile_object in tile_objects:
            self.add_tile(tile_object)


class BoxLid(Component):
    "BoxLid Object, represents the box lid were tiles go before being added to the bag again."
    __slots__ = ('tiles',)
    def __init__(self) -> None:
        super().__init__('BoxLid')
        self.tiles: deque[int] = deque()

    def __repr__(self) -> str:
        return f'BoxLid({self.game})'

    def add_tile(self, tile: Tile) -> None:
        "Add a tile to self."
        if tile.color >= 0 and tile.color < 5:
            self.tiles.append(tile.color)
        else:
            raise Warning(f'BoxLid.add_tile tried to add an invalid tile to self ({tile.color}). Be careful, bad things might be trying to happen.')

    def add_tiles(self, tiles: Iterable[Tile]) -> None:
        "Add multiple tiles to self."
        for tile in tiles:
            self.add_tile(tile)

    def get_tiles(self) -> list[Tile]:
        "Return all tiles in self while deleteing them from self."
        return [Tile(self.tiles.popleft()) for i in range(len(self.tiles))]

    def is_empty(self) -> bool:
        "Return True if self is empty (no tiles on it)."
        return bool(self.tiles)


class Player(MultipartObject):
    "Repesents a player. Made of lots of objects."
    def __init__(
        self,
        game: "Client",
        playerId: int,
        networked: bool = False,
        varient_play: bool = False
    ) -> None:
        "Requires a player Id and can be told to be controled by the network or be in varient play mode."
        super().__init__(f'Player{playerId}')

        self.game = game
        self.pid = playerId
        self.networked = networked
        self.varient_play = varient_play

        self.add_object(Board(self, self.varient_play))
        self.add_object(PatternLine(self))
        self.add_object(FloorLine(self))
        self.add_object(Text('Score_Text', SCOREFONTSIZE, SCORECOLOR))

        self.score = 0
        self.is_turn = False
        self.is_wall_tileing = False
        self.just_held = False
        self.just_dropped = False

##        self.update_score()

        self._lastloc = 0, 0

    def __repr__(self) -> str:
        return 'Player(%r, %i, %s, %s)' % (self.game, self.pid, self.networked, self.varient_play)

    def update_score(self) -> None:
        "Update the scorebox for this player."
        scoreBox = self.get_object_by_name('Score_Text')
        scoreBox.update_value(f'Player {self.pid+1}: {self.score}')

    def turn_now(self) -> None:
        "It is this player's turn now."
        if not self.is_turn:
            pattern_line = self.get_object_by_name('PatternLine')
            if self.is_wall_tileing:
                board = self.get_object_by_name('Board')
                rows = board.get_tile_rows()
                for rowpos in rows:
                    pattern_line.get_row(rowpos).set_background(get_tile_color(rows[rowpos], board.greyshift))
            else:
                pattern_line.set_background(PATSELECTCOLOR)
        self.is_turn = True

    def end_of_turn(self) -> None:
        "It is no longer this player's turn."
        if self.is_turn:
            pattern_line = self.get_object_by_name('PatternLine')
            pattern_line.set_background(None)
        self.is_turn = False

    def now_end(self) -> None:
        "Function called by end state when game is over; Hide pattern lines and floor line."
        pattern = self.get_object_by_name('PatternLine')
        floor = self.get_object_by_name('FloorLine')

        pattern.visible = True
        floor.visible = True

    def reset_position(self) -> None:
        "Reset positions of all parts of self based off self.location."
        x, y = self.location

        bw, bh = self.get_object_by_name('Board').wh
        self.get_object_by_name('Board').location = x+bw/2, y
        lw = self.get_object_by_name('PatternLine').wh[0]/2
        self.get_object_by_name('PatternLine').location = x-lw, y
        fw = self.get_object_by_name('FloorLine').wh[0]
        self.get_object_by_name('FloorLine').location = x-lw*(2/3)+TILESIZE/3.75, y+bh*(2/3)
        self.get_object_by_name('Score_Text').location = x-(bw/3), y-(bh*(2/3))

    def wall_tileing(self) -> None:
        "Do the wall tiling phase of the game for this player."
        self.is_wall_tileing = True
        pattern_line = self.get_object_by_name('PatternLine')
        floorLine = self.get_object_by_name('FloorLine')
        board = self.get_object_by_name('Board')
        boxLid = self.game.get_object_by_name('BoxLid')

        data = pattern_line.wall_tileing()
        boxLid.add_tiles(data['toBox'])
        del data['toBox']

        board.wall_tile_mode(data)

    def done_wall_tileing(self) -> bool:
        "Return True if internal Board is done wall tiling."
        board = self.get_object_by_name('Board')
        return not board.is_wall_tileing()

    def next_round(self) -> None:
        "Called when player is done wall tiling."
        self.is_wall_tileing = False

    def score_phase(self) -> Tile | None:
        "Do the scoring phase of the game for this player."
        board = self.get_object_by_name('Board')
        floorLine = self.get_object_by_name('FloorLine')
        boxLid = self.game.get_object_by_name('BoxLid')
        def saturatescore():
            if self.score < 0:
                self.score = 0

        self.score += board.scoreAdditions()
        self.score += floorLine.scoreTiles()
        saturatescore()

        toBox, number_one = floorLine.get_tiles()
        boxLid.add_tiles(toBox)

        self.update_score()

        return number_one

    def end_client_scoring(self) -> None:
        "Update final score with additional end of game points."
        board = self.get_object_by_name('Board')

        self.score += board.end_score()

        self.update_score()

    def has_horiz_line(self) -> bool:
        "Return True if this player has a horizontal line on their game board filled."
        board = self.get_object_by_name('Board')
        return board.hasFilledRow()

    def get_horizontal_lines(self):
        "Return the number of filled horizontal lines this player has on their game board."
        board = self.get_object_by_name('Board')
        return board.get_filled_rows()

    def process(self, time_passed: float) -> None:
        "Process Player."
        if self.is_turn:# Is our turn?
            if self.visible and self.is_wall_tileing and self.varient_play:
                # If visible, not anymore. Our turn.
                self.visible = False
            if not self.networked:# We not networked.
                cursor = self.game.get_object_by_name('Cursor')
                boxLid = self.game.get_object_by_name('BoxLid')
                pattern_line = self.get_object_by_name('PatternLine')
                floorLine = self.get_object_by_name('FloorLine')
                board = self.get_object_by_name('Board')
                if cursor.is_pressed():# Mouse down?
                    obj, point = self.get_where_touches(cursor.location)
                    if not obj is None and not point is None:# Something pressed
                        if cursor.is_holding():# Cursor holding tiles
                            madeMove = False
                            if not self.is_wall_tileing:# Is wall tiling:
                                if obj == 'PatternLine':
                                    pos, rowNum = point
                                    row = pattern_line.get_row(rowNum)
                                    if not row.is_full():
                                        info = row.get_info(pos)
                                        if not info is None and info.color < 0:
                                            color, held = cursor.get_held_info()
                                            todrop = min(pos+1, row.get_placeable())
                                            tiles = cursor.drop(todrop)
                                            if row.can_place_tiles(tiles):
                                                row.place_tiles(tiles)
                                                madeMove = True
                                            else:
                                                cursor.force_hold(tiles)
                                elif obj == 'FloorLine':
                                    tiles_to_add = cursor.drop()
                                    if floorLine.is_full():# Floor is full,
                                        # Add tiles to box instead.
                                        boxLid.add_tiles(tiles_to_add)
                                    elif floorLine.get_placeable() < len(tiles_to_add):
                                        # Add tiles to floor line and then to box
                                        while len(tiles_to_add) > 0:
                                            if floorLine.get_placeable() > 0:
                                                floorLine.place_tile(tiles_to_add.pop())
                                            else:
                                                boxLid.add_tile(tiles_to_add.pop())
                                    else:# Otherwise add to floor line for all.
                                        floorLine.place_tiles(tiles_to_add)
                                    madeMove = True
                            elif not self.just_held: # Cursor holding and wall tiling
                                if obj == 'Board':
                                    atPoint = board.get_info(point)
                                    if atPoint.color == -6:
                                        column, row = point
                                        cursor_tile = cursor.drop(1)[0]
                                        board_tile = board.get_tile_for_cursor_by_row(row)
                                        if not board_tile is None:
                                            if cursor_tile.color == board_tile.color:
                                                if board.wall_tile_from_point(point):
                                                    self.just_dropped = True
                                                    pattern_line.get_row(row).set_background(None)

                            if madeMove:
                                if not self.is_wall_tileing:
                                    if cursor.holding_number_one:
                                        floorLine.place_tile(cursor.drop_one_tile())
                                    if cursor.get_held_count(True) == 0:
                                        self.game.next_turn()
                        else:# Mouse down, something pressed, and not holding anything
                            if self.is_wall_tileing:# Wall tiling, pressed, not holding
                                if obj == 'Board':
                                    if not self.just_dropped:
                                        columnNum, rowNum = point
                                        tile = board.get_tile_for_cursor_by_row(rowNum)
                                        if not tile is None:
                                            cursor.drag([tile])
                                            self.just_held = True
                else: # Mouse up
                    if self.just_held:
                        self.just_held = False
                    if self.just_dropped:
                        self.just_dropped = False
            if self.is_wall_tileing and self.done_wall_tileing():
                self.next_round()
                self.game.next_turn()
        self.set_attr_all('visible', self.visible)
        super().process(time_passed)

    def get_data(self):
        "Return what makes this Player MultipartObject special."
        data = super().get_data()
        data['pi'] = int(self.pid)
        data['sc'] = f'{self.score:x}'
        data['tu'] = int(self.is_turn)
        data['iwt'] = int(self.is_wall_tileing)
        return data

    def from_data(self, data):
        "Update this Player from data."
        super().from_data()
        self.pid = int(data['pi'])
        self.score = int(data['sc'], 16)
        self.is_turn = bool(data['tu'])
        self.is_wall_tileing = bool(data['iwt'])


class Button(Text):
    "Button Object."
    textcolor = BUTTONTEXTCOLOR
    backcolor = BUTTONBACKCOLOR
    def __init__(self, name: str, min_size: int=10, initValue='', font_size=BUTTONFONTSIZE):
        super().__init__(name, font_size, self.textcolor, background=None)

        self.border_width = math.floor(font_size/12)#5

        self.min_size = min_size
        self.update_value(initValue)

        self.action = lambda: None

        self.add_component(sprite.Click())
        self.add_handler('click', self.handle_click)

    def __repr__(self):
        return f'Button({self.name}, {self.state}, {self.minsize}, {self.font.last_text}, {self.font.pyfont})'

    def bind_action(self, function) -> None:
        "When self is pressed, call given function exactly once. Function takes no arguments."
        self.action = function

    def update_value(self, text: str, size=None, color=None, background='set'):
        disp = text.center(self.min_size)
        super().update_value(f' {disp} ', size, color, background)

        self.font.last_text = disp
        text_rect = self.rect
        w, h = text_rect.size
        extra = self.border_width*2
##        text_rect.size =
        text = self.image
        image = Surface((w+extra, h+extra)).convert_alpha()
        image.fill((0, 0, 0, 0))
        pygame.draw.rect(image, self.backcolor, text_rect, border_radius=20)
        pygame.draw.rect(image, BLACK, text_rect, width=self.border_width, border_radius=20)
        image.blit(text, (self.border_width,self.border_width))
        self.image = image

    async def handle_click(self, event) -> None:
        self.action()

    def from_data(self, data):
        "Update this Button from data."
        super().from_data(data)
        self.update_value(data['flt'])


class ClientState:
    "Base class for all game states."
    name = 'Base Class'
    def __init__(self, name: str) -> None:
        "Initialize state with a name, set self.game to None to be overwritten later."
        self.game = None
        self.name = name

    def __repr__(self):
        return f'<ClientState {self.name}>'

    def entry_actions(self) -> None:
        "Preform entry actions for this ClientState."


    def do_actions(self) -> None:
        "Preform actions for this ClientState."


    def check_state(self) -> str | None:
        "Check state and return new state. None remains in current state."
        return None

    def exit_actions(self) -> None:
        "Preform exit actions for this ClientState."



class MenuState(ClientState):
    "Client State where there is a menu with buttons."
    button_min = 10
    font_size = BUTTONFONTSIZE
    def __init__(self, name: str) -> None:
        "Initialize ClientState and set up self.bh."
        super().__init__(name)
        self.bh = Text.get_font_height(FONT, self.font_size)

        self.next_state: str | None = None

    def add_button(self, name: str, value, action, location='Center', size=font_size, minlen=button_min):
        "Add a new Button object to self.game with arguments. Return button id."
        button = Button(name, minlen, value, size)
        button.bind_action(action)
        if location != 'Center':
            button.location = location
        self.game.add_object(button)
        return button.id

    def add_text(self, name, value, location, color=BUTTONTEXTCOLOR, cx=True, cy=True, size=font_size):
        "Add a new Text object to self.game with arguments. Return text id."
        text = Text(name, font_size=size, color=color, background=None, cx=cx, cy=cy)
        text.location = location
        text.update_value(value)
        self.game.add_object(text)
        return text.id

    def entry_actions(self):
        "Clear all objects, add cursor object, and set up to_state."
        self.next_state = None

        self.game.rm_star()
        self.game.add_object(Cursor(self.game))

    def set_var(self, attribute, value):
        "Set MenuState.{attribute} to {value}."
        setattr(self, attribute, value)

    def to_state(self, state_name: str) -> Callable[[], None]:
        "Return a function that will change game state to state_name."
        def to_state_name():
            f"Set MenuState.to_state to {state_name}."
            self.next_state = state_name
        return to_state_name

    def var_dependant_to_state(self, **kwargs):
        "attribute name = (target value, on trigger tostate)."
        for state in kwargs:
            if not len(kwargs[state]) == 2:
                raise ValueError(f'Key "{state}" is invalid!')
            key, value = kwargs[state]
            if not hasattr(self, key):
                raise ValueError(f'{self} object does not have attribute "{key}"!')
        def to_state_by_attributes():
            "Set MenuState.to_state to a new state if conditions are right."
            for state in kwargs:
                key, value = kwargs[state]
                if getattr(self, key) == value:
                    self.next_state = state
        return to_state_by_attributes

    def with_update(self, update_function):
        "Return a wrapper for a function that will call update_function after function."
        def update_wrapper(function):
            "Wrapper for any function that could require a screen update."
            @wraps(function)
            def function_with_update():
                "Call main function, then update function."
                function()
                update_function()
            return function_with_update
        return update_wrapper

    def update_text(self, text_name, value_func):
        "Update text object with text_name's display value."
        def updater():
            f"Update text object {text_name}'s value with {value_func}."
            text = self.game.get_object_by_name(text_name)
            text.update_value(value_func())
        return updater

    def toggle_button_state(self, text_name, bool_attr, text_func):
        "Return function that will toggle the value of text object <text_name>, toggleing attribute <bool_attr>, and setting text value with text_func."
        def valfunc():
            "Return the new value for the text object. Gets called AFTER value is toggled."
            return text_func(getattr(self, bool_attr))
        @self.with_update(self.update_text(text_name, valfunc))
        def toggleValue():
            "Toggle the value of bool_attr."
            self.set_var(bool_attr, not getattr(self, bool_attr))
        return toggleValue

    def check_state(self) -> str | int:
        "Return self.to_state."
        return self.next_state


class InitState(ClientState):
    __slots__ = ()
    def __init__(self) -> None:
        super().__init__('Init')

    def check_state(self) -> str:
        return 'Title'


class TitleScreen(MenuState):
    "Client state when the title screen is up."
    def __init__(self) -> None:
        super().__init__('Title')

    def raise_close(self) -> None:
        pygame.event.post(pygame.event(QUIT))

    def entry_actions(self) -> None:
        super().entry_actions()
        sw, sh = SCREENSIZE
        self.add_button('ToSettings', 'New Client', self.to_state('Settings'), (sw/2, sh/2-self.bh*0.5))
        self.add_button('ToCredits', 'Credits', self.to_state('Credits'), (sw/2, sh/2+self.bh*3), self.font_size/1.5)
        self.add_button('Quit', 'Quit', self.raise_close, (sw/2, sh/2+self.bh*4), self.font_size/1.5)


class CreditsScreen(MenuState):
    "Client state when credits for original game are up."
    def __init__(self) -> None:
        super().__init__('Credits')

    def entry_actions(self) -> None:
        super().entry_actions()

    def check_state(self) -> str:
        return 'Title'


class SettingsScreen(MenuState):
    "Client state when user is defining game type, players, etc."
    def __init__(self):
        super().__init__('Settings')

        self.player_count = 0#2
        self.host_mode = True
        self.variant_play = False

    def entry_actions(self):
        "Add cursor object and tons of button and text objects to the game."
        super().entry_actions()

        def add_numbers(start, end, width_each, cx, cy):
            count = end-start+1
            evencount = count % 2 == 0
            mid = count//2
            def addNumber(number, display, cx, cy):
                if evencount:
                    if number < mid:
                        x = number-start-0.5
                    else:
                        x = number-mid+0.5
                else:
                    if number < mid:
                        x = number-start+1
                    elif number == mid:
                        x = 0
                    else:
                        x = number-mid

                @self.with_update(self.update_text('Players', lambda: f'Players: {self.player_count}'))
                def set_player_count():
                    f"Set varibable player_count to {display} while updating text."
                    return self.set_var('player_count', display)

                self.add_button(f'SetCount{number}', str(display), set_player_count,
                               (cx+(width_each*x), cy), size=self.font_size/1.5, minlen=3)
            for i in range(count):
                addNumber(i, start+i, cx, cy)

        sw, sh = SCREENSIZE
        cx = sw/2
        cy = sh/2

        host_text = lambda x: f'Host Mode: {x}'
        self.add_text('Host', host_text(self.host_mode), (cx, cy-self.bh*3))
        self.add_button('ToggleHost', 'Toggle', self.toggle_button_state('Host', 'host_mode', host_text), (cx, cy-self.bh*2), size=self.font_size/1.5)

        # TEMPORARY: Hide everything to do with "Host Mode", networked games arn't done yet.
        self.game.set_attr_all('visible', True)

        varient_text = lambda x: f'Varient Play: {x}'
        self.add_text('Varient', varient_text(self.variant_play), (cx, cy-self.bh))
        self.add_button('ToggleVarient', 'Toggle', self.toggle_button_state('Varient', 'variant_play', varient_text), (cx, cy), size=self.font_size/1.5)

        self.add_text('Players', f'Players: {self.player_count}', (cx, cy+self.bh))
        add_numbers(2, 4, 70, cx, cy+self.bh*2)

        var_to_state = self.var_dependant_to_state(FactoryOffer=('host_mode', True), FactoryOfferNetworked=('host_mode', False))
        self.add_button('StartClient', 'Start Client', var_to_state, (cx, cy+self.bh*3))

    def exit_actions(self) -> None:
        self.game.start_game(self.player_count, self.variant_play, self.host_mode)
        self.game.bag.full_reset()


class PhaseFactoryOffer(ClientState):
    "Client state when it's the Factory Offer Stage."
    def __init__(self):
        super().__init__('FactoryOffer')

    def entry_actions(self):
        "Advance turn."
        self.game.next_turn()

    def check_state(self):
        "If all tiles are gone, go to wall tiling. Otherwise keep waiting for that to happen."
        fact = self.game.get_object_by_name('Factories')
        table = self.game.get_object_by_name('TableCenter')
        cursor = self.game.get_object_by_name('Cursor')
        if fact.is_all_empty() and table.is_empty() and not cursor.is_holding(True):
            return 'WallTiling'
        return None


class PhaseWallTiling(ClientState):
    def __init__(self):
        super().__init__('WallTiling')

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
        if not self.not_processed:
            return
        if self.game.player_turn in self.not_processed:
            player = self.game.get_player(self.game.player_turn)
            if player.done_wall_tileing():
                # Once player is done wall tiling, score their moves.
                number_one = player.score_phase()#Also gets if they had the number one tile.
                if number_one:
                    # If player had the number one tile, remember that.
                    self.next_starter = self.game.player_turn
                    # Then, add the number one tile back to the table center.
                    table = self.game.get_object_by_name('TableCenter')
                    table.add_number_one_tile()
                # After calculating their score, delete player from un-processed list
                self.not_processed.remove(self.game.player_turn)
                # and continue to the next un-processed player.
                self.game.next_turn()
        else:
            self.game.next_turn()

    def check_state(self):
        cursor = self.game.get_object_by_name('Cursor')
        if not self.not_processed and not cursor.is_holding():
            return 'PrepareNext'
        return None

    def exit_actions(self):
        # Set up the player that had the number one tile to be the starting player next round.
        self.game.player_turn_over()
        # Goal: make (self.player_turn + 1) % self.players = self.next_starter
        nturn = self.next_starter - 1
        if nturn < 0:
            nturn += self.game.players
        self.game.player_turn = nturn


class PhasePrepareNext(ClientState):
    __slots__ = ('new_round',)
    def __init__(self) -> None:
        super().__init__('PrepareNext')

    def entry_actions(self) -> None:
        players = (self.game.get_player(pid) for pid in range(self.game.players))
        complete = (player.has_horiz_line() for player in players)
        self.new_round = not any(complete)

    def do_actions(self) -> None:
        if self.new_round:
            fact = self.game.get_object_by_name('Factories')
            # This also handles bag re-filling from box lid.
            fact.divy_up_tiles()

    def check_state(self) -> str:
        if self.new_round:
            return 'FactoryOffer'
        return 'End'


class EndScreen(MenuState):
    def __init__(self) -> None:
        super().__init__('End')
        self.ranking = {}
        self.wininf = ''

    def get_winners(self):
        "Update self.ranking by player scores."
        self.ranking = {}
        scpid = {}
        for pid in range(self.game.players):
            player = self.game.get_player(pid)
            player.now_end()
            if not player.score in scpid:
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
                lines = [(p.get_horizontal_lines(), p.pid) for p in players]
                last = None
                for c, pid in sorted(lines, key=lambda x: x[0], reverse=True):
                    if last == c:
                        self.ranking[rank-1] += [pid + 1]
                        continue
                    last = c
                    self.ranking[rank] = [pid + 1]
                    rank += 1
            else:
                self.ranking[rank] = [pids[0] + 1]
                rank += 1
        # Finally, make nice text.
        text = ''
        for rank in sorted(self.ranking):
            line = 'Player'
            players = self.ranking[rank]
            cnt = len(players)
            if cnt > 1:
                line += 's'
            line += ' '
            if cnt == 1:
                line += '{}'
            elif cnt == 2:
                line += '{} and {}'
            elif cnt >= 3:
                tmp = (['{}'] * (cnt-1)) + ['and {}']
                line += ', '.join(tmp)
            line += ' '
            if cnt == 1:
                line += 'got'
            else:
                line += 'tied for'
            line += ' '
            if rank <= 2:
                line += ('1st', '2nd')[rank-1]
            else:
                line += f'{rank}th'
            line += ' place!\n'
            text += line.format(*players)
        self.wininf = text[:-1]

    def entry_actions(self):
        # Figure out who won the game by points.
        self.get_winners()
        # Hide everything
        table = self.game.get_object_by_name('TableCenter')
        table.visible = True

        fact = self.game.get_object_by_name('Factories')
        fact.set_attr_all('visible', True)

        # Add buttons
        bid = self.add_button('ReturnTitle', 'Return to Title', self.to_state('Title'), (SCREENSIZE[0]/2, math.floor(SCREENSIZE[1]*(4/5))))
        buttontitle = self.game.get_object(bid)
        buttontitle.Render_Priority = 'last-1'
        buttontitle.cur_time = 2

        # Add score board
        x = SCREENSIZE[0]/2
        y = 10
        idx = 0
        for line in self.wininf.split('\n'):
            lid = self.add_text(f'Line{idx}', line, (x, y), cx=True, cy=False)
##            self.game.get_object(bid).Render_Priority = f'last{-(2+idx)}'
            self.game.get_object(bid).Render_Priority = f'last-2'
            idx += 1
            y += self.bh


class Client(ObjectHandler):
    "Client object, contains most of what's required for Azul."
    tile_size = 30
    def __init__(self):
        super().__init__('Client')

        self.states = {}
        self.active_state = None

        self.add_states([InitState(),
                         TitleScreen(),
                         CreditsScreen(),
                         SettingsScreen(),
                         PhaseFactoryOffer(),
                         PhaseWallTiling(),
                         PhasePrepareNext(),
                         EndScreen(),
                         ])
        self.initialized_state = False

        self.is_host = True
        self.players = 0
        self.factories = 0

        self.player_turn = 0

        # Tiles
        self.bag = Bag(TILECOUNT, REGTILECOUNT)

        ## New
        self.add_handler('KeyUp', self.handle_keyup)
        self.add_handler('tick', self.handle_tick)

    async def handle_keyup(self, event: Event[int]) -> str | None:
        "If escape key let go, post quit event"
        if event['key'] == K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            return 'break'
        return None

    def __repr__(self):
        return 'Client()'

    async def __call__(self, event: Event[Any]) -> None:
        await super().__call__(event)
        if event.name == 'tick':
            return
        print(event)

    def screenshot(self) -> tuple[str, str]:
        "Save a screenshot of this game's most recent frame."
        surface = pygame.surface.Surface(SCREENSIZE)
        self.render(surface)
        str_time = '-'.join(time.asctime().split(' '))
        filename = f'Screenshot_at_{str_time}.png'

        if not os.path.exists('Screenshots'):
            os.mkdir('Screenshots')

        surface.unlock()
        pygame.image.save(surface, os.path.join('Screenshots', filename),
                          filename)
        del surface

        savepath = os.path.join(os.getcwd(), 'Screenshots')

        print(f'Saved screenshot as "{filename}" in "{savepath}".')
        return savepath, filename

    def add_states(self, states):
        "Add game states to self."
        for state in states:
            if not isinstance(state, ClientState):
                raise ValueError(f'"{state}" Object is not a subclass of ClientState!')
            state.game = self
            self.states[state.name] = state

    def set_state(self, new_state_name):
        "Change states and preform any exit / entry actions."
        # Ensure the new state is valid.
        if not new_state_name in self.states:
            raise ValueError(f'State "{new_state_name}" does not exist!')

        # If we have an active state,
        if not self.active_state is None:
            # Preform exit actions
            self.active_state.exit_actions()

        # The active state is the new state
        self.active_state = self.states[new_state_name]
        # Preform entry actions for new active state
        self.active_state.entry_actions()

    def update_state(self):
        "Preform the actions of the active state and potentially change states."
        # Only continue if there is an active state
        if self.active_state is None:
            return

        # Preform the actions of the active state and check conditions
        self.active_state.do_actions()

        new_state_name = self.active_state.check_state()
        if not new_state_name is None:
            self.set_state(new_state_name)

    def add_object(self, obj):
        "Add an object to the game."
        try:
            setattr(obj, 'game', self)
        except AttributeError:
            pass
        super().add_object(obj)

    async def handle_tick(self, event: Event[float]) -> None:
        "Process all the objects and self."
        if not self.initialized_state:
            self.set_state('Init')
            self.initialized_state = True
        self.update_state()

    def get_player(self, pid):
        "Get the player with player id pid."
        if self.players:
            return self.get_object_by_name(f'Player{pid}')
        raise RuntimeError('No players!')

    def player_turn_over(self):
        "Call end_of_turn for current player."
        if self.player_turn >= 0 and self.player_turn < self.players:
            old_player = self.get_player(self.player_turn)
            if old_player.is_turn:
                old_player.end_of_turn()

    def next_turn(self):
        "Tell current player it's the end of their turn, and update who's turn it is and now it's their turn."
        if self.is_host:
            self.player_turn_over()
            last = self.player_turn
            self.player_turn = (self.player_turn + 1) % self.players
            if self.player_turn == last and self.players > 1:
                self.next_turn()
                return
            new_player = self.get_player(self.player_turn)
            new_player.turn_now()

    def start_game(self, players, varient_play=False, host_mode=True, address=''):
        "Start a new game."
        maxPlayers = 4
        self.players = saturate(players, 1, maxPlayers)
        self.is_host = host_mode
        self.factories = self.players * 2 + 1

        self.rm_star()

        self.add_object(Cursor(self))
        self.add_object(TableCenter(self))
        self.add_component(BoxLid())

        if self.is_host:
            self.bag.reset()
            self.player_turn = random.randint(-1, self.players-1)
        else:
            self.player_turn = 'Unknown'

        cx, cy = SCREENSIZE[0]/2, SCREENSIZE[1]/2
        out = math.sqrt(cx ** 2 + cy ** 2) // 3 * 2

        mdeg = 360 // maxPlayers

        for pid in range(self.players):
            networked = False
            newp = Player(self, pid, networked, varient_play)

            truedeg = (self.players + 1 - pid) * (360 / self.players)
            closedeg = truedeg // mdeg * mdeg + 45
            rad = math.radians(closedeg)

            newp.location = round(cx+out*math.sin(rad)), round(cy+out*math.cos(rad))
            self.add_object(newp)
        if self.is_host:
            self.next_turn()

        factory = Factories(self, self.factories)
        factory.location = cx, cy
        self.add_object(factory)
        self.process_objects(0)

        if self.is_host:
            self.next_turn()

def as_component_event(event: pygame.event.Event) -> Event[str]:
    "Convert pygame event to component event"
    if event.type > USEREVENT:
        raise_event = Event(f'Custom_Event_{event.type-USEREVENT}', event.dict)
        print(raise_event)
        return raise_event
    return Event(pygame.event.event_name(event.type), event.dict)

async def async_run() -> None:
    "Start program"
    # Set up the screen
    screen = pygame.display.set_mode(tuple(SCREENSIZE),
                                     RESIZABLE, 16, vsync=VSYNC)
    pygame.display.set_caption(f'{__title__} v{__version__}')
##    pygame.display.set_icon(pygame.image.load('icon.png'))
    pygame.display.set_icon(get_tile_image(Tile(5), 32))

    MUSIC_END = USEREVENT + 1#This event is sent when a music track ends

    # Set music end event to our new event
    pygame.mixer.music.set_endevent(MUSIC_END)

##    # Load and start playing the music
##    pygame.mixer.music.load('sound/Captain Scurvy.mp3')
####    pygame.mixer.music.load('sound/Jaunty Gumption.mp3')
##    pygame.mixer.music.play()

    group = Client()

    await group(Event('__init__'))

    screen.fill(BACKGROUND)
    group.clear(screen, screen.copy().convert())
    group.set_timing_treshold(1000/FPS)

    running = True

    # Set up the FPS clock
##    clock = pygame.time.Clock()
    clock = Clock()

    # While the game is active
    while running:
        # Event handler
        async with trio.open_nursery() as nursery:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MUSIC_END:
                    # If the music ends, stop it and play it again.
                    pygame.mixer.music.stop()
                    pygame.mixer.music.play()
                nursery.start_soon(group, as_component_event(event))

        # Get the time passed from the FPS clock
        time_passed = await clock.tick(FPS)

        # Update the display
        rects = group.draw(screen)
        pygame.display.update(rects)

        await group(Event('tick',
            time_passed = time_passed / 1000,
            fps = clock.get_fps(),
        ))
    # Once the game has ended, stop the music
    pygame.mixer.music.stop()


def run() -> None:
    "Synchronous entry point"
    trio.run(async_run)

def save_crash_img() -> None:
    "Save the last frame before the game crashed."
    surface = pygame.display.get_surface().copy()
    str_time = '-'.join(time.asctime().split(' '))
    filename = f'Crash_at_{str_time}.png'

    if not os.path.exists('Screenshots'):
        os.mkdir('Screenshots')

##    surface.lock()
    pygame.image.save(surface, os.path.join('Screenshots', filename),
                      filename)
##    surface.unlock()
    del surface

    savepath = os.path.join(os.getcwd(), 'Screenshots')

    print(f'Saved screenshot as "{filename}" in "{savepath}".')

if __name__ == '__main__':
    # Linebreak before, as pygame prints a message on import.
    print(f'\n{__title__} v{__version__}\nProgrammed by {__author__}.')
    try:
        # Initialize Pygame
        fails = pygame.init()[1]
        if fails > 0:
            print('Warning! Some modules of Pygame have not initialized properly!')
            print('This can occur when not all required modules of SDL, which pygame utilizes, are installed.')
        run()
##    except BaseException as ex:
##        reraise = True#False
##
####        print('Debug: Activiting Post motem.')
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
