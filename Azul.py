#!/usr/bin/env python3
# Azul board game clone, now on the computer!
# -*- coding: utf-8 -*-

# Programmed by CoolCat467

import os
import random
import time
import math
from collections import deque, Counter
from functools import wraps, lru_cache

from numpy import array

TEXTPYGAMEIMPORTERROR = False

# If pygame is not installed, install it
try:
    from pygame.locals import *
    import pygame
except ImportError:
    if not TEXTPYGAMEIMPORTERROR:
        try:
            import errorbox
        except ImportError:
            TEXTPYGAMEIMPORTERROR = True
        else:
            errorbox.errorbox('Error: Pygame Not Installed', 'Please install Pygame by\nrunning the command\n"pip3 install pygame"\nin your computer\'s command\nprompt/terminal/shell.')
    if TEXTPYGAMEIMPORTERROR:
        print('\nError: Pygame is not installed!\n')#, file=os.sys.stderr)
        while True:
            inp = input('Would you like to attempt to install Pygame automatically? (y/n) : ').lower()
            if inp in ('y', 'n'):
                break
            else:
                print('Please enter a valid answer.\n')
        if inp == 'y':
            print('\nAttempting to automatically install pygame...\n')
            out = os.system('pip3 install pygame --user')
            if out == 0:
                print('Pygame installed sucessfully!\nPlease Restart the program.\n\nNote: If you get this message multiple times, something may be broken and you might have to manually install pygame instead.')
            else:
                print('Something went wrong installing pygame.', file=os.sys.stderr)
                inp = 'n'
        if inp == 'n':
            print('\nTo manually install pygame, open your computer\'s command prompt/terminal/shell')
            print('and type in the command "pip3 install pygame --user".')
        input('\nPress Enter to Continue. ')
    os.abort()


__title__ = 'Azul'
__author__ = 'CoolCat467'
__version__ = '0.0.0'
__ver_major__ = 0
__ver_minor__ = 0
__ver_patch__ = 0

SCREENSIZE = (650, 600)
FPS = 30
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
# Pygame Version
PYGAME_VERSION = int(''.join(x for x in pygame.ver.split('.') if x.isdigit()))
# Game stuff
# Tiles
TILECOUNT = 100
REGTILECOUNT = 5
TILECOLORS = (BLUE, YELLOW, RED, BLACK, CYAN, (WHITE, BLUE))
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

@lru_cache()
def mkSquare(color, size):
    """Return a surface of a square of given color and size."""
    s = int(size)
    surf = pygame.Surface((s, s))
    surf.fill(color)
    return surf

def outlineRectangle(surface, color, percent=0.1):
    """Return a surface after adding an outline of given color. Percentage is how big the outline is."""
    w, h = surface.get_size()
    inside_surf = pygame.transform.scale(surface.copy(), (round(w*(1 - percent)), round(h*(1 - percent))))
    surface.fill(color)
    surface.blit(inside_surf, (math.floor(w * percent / 2), math.floor(h * percent / 2)))
    return surface

def autoCropClear(surface, clear=(0, 0, 0, 0)):
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
    column = lambda x: (surface.get_at((x, y)) for y in range(h))
    row    = lambda y: (surface.get_at((x, y)) for x in range(w))
    leftc  = find_end(column, range(w))
    rightc = find_end(column, range(w-1, -1, -1))
    topc   = find_end(row, range(h))
    floorc = find_end(row, range(h-1, -1, -1))
    surface.unlock()
    dim = pygame.rect.Rect(leftc, topc, rightc-leftc, floorc-topc)
    return surface.subsurface(dim)

@lru_cache()
def getTileColor(tileColor, greyshift=GREYSHIFT):
    """Return the color a given tile should be."""
    if tileColor < 0:
        if tileColor == -6:
            return GREY
        return lerpColor(TILECOLORS[abs(tileColor+1)], GREY, greyshift)
    elif tileColor < 5:
        return TILECOLORS[tileColor]
    elif tileColor >= 5:
        raise ValueError('Cannot properly return tile colors greater than five!')

@lru_cache()
def getTileSymbolAndColor(tileColor, greyshift=GREYSHIFT):
    """Return the color a given tile should be."""
    if tileColor < 0:
        if tileColor == -6:
            return ' ', GREY
        symbol, scolor = TILESYMBOLS[abs(tileColor+1)]
        return symbol, lerpColor(scolor, GREY, greyshift)
    elif tileColor <= 5:
        return TILESYMBOLS[tileColor]
    elif tileColor >= 6:
        raise ValueError('Cannot properly return tile colors greater than five!')

def addSymbolToTileSurf(surf, tilecolor, tilesize, greyshift=GREYSHIFT, font=FONT):
    symbol, scolor = getTileSymbolAndColor(tilecolor, greyshift)
    pyfont = pygame.font.Font(font, math.floor(math.sqrt(tilesize**2*2))-1)
    
    symbolsurf = pyfont.render(symbol, True, scolor)
    symbolsurf = autoCropClear(symbolsurf)
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

@lru_cache()
def getTileImage(tile, tilesize, greyshift=GREYSHIFT, outlineSize=0.2, font=FONT):
    "Return a surface of a given tile."
    cid = tile.color
    if cid < 5:
        color = getTileColor(cid, greyshift)
        
    elif cid >= 5:
        color, outline = TILECOLORS[cid]
        surf = outlineRectangle(mkSquare(color, tilesize), outline, outlineSize)
        # Add tile symbol
        addSymbolToTileSurf(surf, cid, tilesize, greyshift, font)
        
        return surf
    surf = mkSquare(color, tilesize)
    # Add tile symbol
##    addSymbolToTileSurf(surf, cid, tilesize, greyshift, font)
    
    return surf

def setAlpha(surface, alpha):
    """Return a surface by replacing the alpha chanel of it with given alpha value, preserve color."""
    surface = surface.copy().convert_alpha()
    w, h = surface.get_size()
    for y in range(h):
        for x in range(w):
            r, g, b = surface.get_at((x, y))[:3]
            surface.set_at((x, y), pygame.Color(r, g, b, alpha))
    return surface

@lru_cache()
def getTileContainerImage(wh, back):
    """Return a tile container image from a width and a heigth and a background color, and use a game's cache to help."""
    image = pygame.surface.Surface(wh)
    image.convert_alpha()
    image = setAlpha(image, 0)
    
    if not back is None:
        image.convert()
        image.fill(back)
    return image

class Font(object):
    """Font object, simplify using text."""
    def __init__(self, fontName, fontsize=20, color=(0, 0, 0), cx=True, cy=True, antialias=False, background=None, doCache=True):
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
        return 'Font(%r, %i, %r, %r, %r, %r, %r, %r)' % (self.font, self.size, self.color, self.center[0], self.center[1], self.antialias, self.background, self.doCache)
    
    def _changeFont(self):
        """Set self.pyfont to a new pygame.font.Font object from data we have."""
        self.pyfont = pygame.font.Font(self.font, self.size)
    
    def _cache(self, surface):
        """Set self.cache to surface"""
        self.cache = surface
    
    def get_height(self):
        """Return the height of font."""
        return self.pyfont.get_height()
    
    def render_nosurf(self, text, size=None, color=None, background=None, forceUpdate=False):
        """Render and return a surface of given text. Use stored data to render, if arguments change internal data and render."""
        updateCache = self.cache is None or forceUpdate or text != self.lastText
        # Update internal data if new values given
        if not size is None:
            self.size = int(size)
            self._changeFont()
            updateCache = True
        if not color is None:
            self.color = color
            updateCache = True
        if self.background != background:
            self.background = background
            updateCache = True
        
        if self.doCache:
            if updateCache:
                self.lastText = text
                surf = self.pyfont.render(text, self.antialias, self.color, self.background).convert_alpha()
                self._cache(surf.copy())
            else:
                surf = self.cache
        else:
            # Render the text using the pygame font
            surf = self.pyfont.render(text, self.antialias, self.color, self.background).convert_alpha()
        return surf
    
    def render(self, surface, text, xy, size=None, color=None, background=None, forceUpdate=False):
        """Render given text, use stored data to render, if arguments change internal data and render."""
        surf = self.render_nosurf(text, size, color, background, forceUpdate)
        
        if True in self.center:
            x, y = xy
            cx, cy = self.center
            w, h = surf.get_size()
            if cx:
                x -= w/2
            if cy:
                y -= h/2
            xy = (int(x), int(y))
        
        surface.blit(surf, xy)
    pass

class ObjectHandler(object):
    """ObjectHandler class, ment to be used for other classes."""
    def __init__(self):
        self.objects = {}
        self.nextId = 0
        self.cache = {}
        
        self.recalculate_render = True
        self._render_order = ()
    
    def add_object(self, obj:object):
        """Add an object to the game."""
        obj.id = self.nextId
        self.objects[self.nextId] = obj
        self.nextId += 1
        self.recalculate_render = True
    
    def rm_object(self, obj:object):
        """Remove an object from the game."""
        del self.objects[obj.id]
        self.recalculate_render = True
    
    def rm_star(self):
        """Remove all objects from self.objects."""
        for oid in list(self.objects):
            self.rm_object(self.objects[oid])
        self.nextId = 0
    
    def get_object(self, objectId:int):
        """Return the object accociated with object id given. Return None if object not found."""
        if objectId in self.objects:
            return self.objects[objectId]
        return None
    
    def get_objects_with_attr(self, attribute:str):
        """Return a tuple of object ids with given attribute."""
        return tuple((oid for oid in self.objects if hasattr(self.objects[oid], attribute)))
    
    def get_object_by_attr(self, attribute:str, value):
        """Return a tuple of object ids with <attribute> that are equal to <value>."""
        matches = []
        for oid in self.get_objects_with_attr(attribute):
            if getattr(self.objects[oid], attribute) == value:
                matches.append(oid)
        return tuple(matches)
    
    def get_object_by_name(self, name:str):
        """Returns a tuple of object ids with names matching <name>."""
        return self.get_object_by_attr('name', name)
    
    def reset_cache(self):
        """Reset the cache."""
        self.cache = {}
    
    def getObjectByName(self, objName):
        """Get object by name, with cache."""
        if not objName in self.cache:
            ids = self.get_object_by_name(objName)
            if ids:
                self.cache[objName] = min(ids)
            else:
                raise RuntimeError(f'{objName} Object Not Found!')
        return self.get_object(self.cache[objName])
    
    def set_attr_all(self, attribute:str, value):
        """Set given attribute in all of self.objects to given value in all objects with that attribute."""
        for oid in self.get_objects_with_attr(attribute):
            setattr(self.objects[oid], attribute, value)
    
    def recalculateRenderOrder(self):
        """Recalculate the order in which to render objects to the screen."""
        new = {}
        cur = 0
        for oid in reversed(self.objects):
            obj = self.objects[oid]
            if hasattr(obj, 'RenderPriority'):
                prior = getattr(obj, 'RenderPriority')
                if isinstance(prior, str):
                    add = 0
                    if prior[:4] == 'last':
                        add = prior[4:] or 0
                        try:
                            add = int(add)
                        except ValueError:
                            add = 0
                        pos = len(self.objects)+add
                    if prior[:5] == 'first':
                        add = prior[5:] or 0
                        try:
                            add = int(add)
                        except ValueError:
                            add = 0
                        pos = -1+add
                    if not pos in new.values():
                        new[oid] = pos
                    else:
                        while True:
                            if add < 0:
                                pos -= 1
                            else:
                                pos += 1
                            if not pos in new.values():
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
        revnew = {new[k]:k for k in new}
        new = []
        for key in sorted(revnew):
            new.append(revnew[key])
        self._render_order = tuple(new)
    
    def processObjects(self, time_passed:float):
        """Call the process function on all objects."""
        if self.recalculate_render:
            self.recalculateRenderOrder()
            self.recalculate_render = False
        for oid in iter(self.objects):
            self.objects[oid].process(time_passed)
    
    def renderObjects(self, surface):
        """Render all objects to surface."""
        if not self._render_order or self.recalculate_render:
            self.recalculateRenderOrder()
            self.recalculate_render = False
        for oid in self._render_order:#reversed(list(self.objects.keys())):
            self.objects[oid].render(surface)
    
    def __del__(self):
        self.reset_cache()
        self.rm_star()
    pass

class Object(object):
    """Object object."""
    name = 'Object'
    def __init__(self, name):
        """Sets self.name to name, and other values for rendering.
            
           Defines the following attributes:
            self.name
            self.image
            self.location
            self.wh
            self.hidden
            self.locModOnResize
            self.id"""
        self.name = str(name)
        self.image = None
        self.location = Vector2(round(SCREENSIZE[0]/2), round(SCREENSIZE[1]/2))
        self.wh = 0, 0
        self.hidden = False
        self.locModOnResize = 'Scale'
        self.scLast = SCREENSIZE
        
        self.id = 0
    
    def __repr__(self):
        """Return {self.name}()."""
        return f'{self.name}()'
    
    def getImageZero_noFix(self):
        """Return the screen location of the topleft point of self.image."""
        return self.location[0]-self.wh[0]/2, self.location[1]-self.wh[1]/2
    
    def getImageZero(self):
        """Return the screen location of the topleft point of self.image fixed to intiger values."""
        x, y = self.getImageZero_noFix()
        return int(x), int(y)
    
    def getRect(self):
        """Return a Rect object representing this Object's area."""
        return Rect(self.getImageZero(), self.wh)
    
    def pointIntersects(self, screenLocation):
        """Return True if this Object intersects with a given screen location."""
        return self.getRect().collidepoint(screenLocation)
    
    def toImageSurfLoc(self, screenLocation):
        """Return the location a screen location would be at on the objects image. Can return invalid data."""
        # Get zero zero in image locations
        zx, zy = self.getImageZero()#Zero x and y
        sx, sy = screenLocation#Screen x and y
        return sx - zx, sy - zy#Location with respect to image dimentions
    
    def process(self, time_passed):
        """Process Object. Replace when calling this class."""
        pass
    
    def render(self, surface):
        """Render self.image to surface if self.image is not None. Updates self.wh."""
        if self.image is None or self.hidden:
            return
        self.wh = self.image.get_size()
        x, y = self.getImageZero()
        surface.blit(self.image, (int(x), int(y)))
##        pygame.draw.rect(surface, MAGENTA, self.getRect(), 1)
    
    def __del__(self):
        """Delete self.image"""
        del self.image
    
    def screen_size_update(self):
        """Function called when screensize is changed."""
        nx, ny = self.location
        
        if self.locModOnResize == 'Scale':
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
        data['x'] = round(x)
        data['y'] = round(y)
        data['hid'] = int(self.hidden)
        data['id'] = int(self.id)
        return data
    
    def from_data(self, data):
        """Update an object using data."""
        self.location = float(data['x']), float(data['y'])
        self.hidden = bool(data['hid'])
        self.id = int(data['id'])
    pass

class MultipartObject(Object, ObjectHandler):
    """Thing that is both an Object and an ObjectHandler, and is ment to be an Object made up of multiple Objects."""
    def __init__(self, name):
        """Initalize Object and ObjectHandler of self.
           
           Also set self._lastloc and self._lasthidden to None"""
        Object.__init__(self, name)
        ObjectHandler.__init__(self)
        
        self._lastloc = None
        self._lasthidden = None
    
    def resetPosition(self):
        """Reset the position of all objects within."""
        raise NotImplemented
    
    def getWhereTouches(self, point):
        """Return where a given point touches in self. Returns (None, None) with no intersections."""
        for oid in self.objects:
            obj = self.objects[oid]
            if hasattr(obj, 'getTilePoint'):
                output = obj.getTilePoint(point)
                if not output is None:
                    return obj.name, output
            else:
                raise Warning('Not all of self.objects have the getTilePoint attribute!')
        return None, None
    
    def process(self, time_passed):
        """Process Object self and ObjectHandler self and call self.resetPosition on location change."""
        Object.process(self, time_passed)
        ObjectHandler.processObjects(self, time_passed)
        
        if self.location != self._lastloc:
            self.resetPosition()
            self._lastloc = self.location
        
        if self.hidden != self._lasthidden:
            self.set_attr_all('hidden', self.hidden)
            self._lasthidden = self.hidden
    
    def render(self, surface):
        """Render self and all parts to the surface."""
        Object.render(self, surface)
        ObjectHandler.renderObjects(self, surface)
    
    def get_data(self):
        """Return what makes this MultipartObject special."""
        data = super().get_data()
        data['objs'] = tuple([self.objects[oid].get_data() for oid in self.objects])
        return data
    
    def from_data(self, data):
        """Update this MultipartObject from data."""
        super().from_data(self)
        for objdata in data['objs']:
            self.objects[int(objdata['id'])].from_data(objdata)
    
    def __del__(self):
        Object.__del__(self)
        ObjectHandler.__del__(self)
    pass

class NerworkServer(object):
    """NetworkServer Class, job is to talk to connect classes over the interwebs."""
    def __init__(self, port):
        self.name = 'NetworkServer'
    
##    def add_client(self)
    pass

class NetworkClient(object):
    """NetworkClient Class, job is to talk to NetworkServer and therefore other NetworkClient classes over the interwebs."""
    def __init__(self, ip_address, port):
        self.name = 'NetworkClient'
    
    def requestData(self, dataName):
        """Request a certain field of information from the server."""
        pass
    
    pass

class Tile(object):
    """Represents a Tile."""
    def __init__(self, color):
        """Needs a color value, or this is useless."""
        self.color = color
    
    def __repr__(self):
        return 'Tile(%i)' % self.color
    
    def get_data(self):
        """Return self.color"""
        return f'T[{self.color}]'
    
    @classmethod
    def from_data(cls, data):
        """Return a new Tile object using data."""
        return cls.__init__(int(data[2:-1]))
    pass

class TileRenderer(Object):
    """Base class for all objects that need to render tiles."""
    greyshift = GREYSHIFT
    tileSize = TILESIZE
    def __init__(self, name, game, tileSeperation='Auto', background=TILEDEFAULT):
        """Initialize renderer. Needs a game object for its cache and optional tile seperation value and background RGB color.

           Defines the following attributes during initialization and uses throughout:
            self.game
            self.wh
            self.tileSep
            self.tileFull
            self.back
            and finally, self.imageUpdate
           
           The following functions are also defined:
            self.clearImage
            self.renderTile
            self.updateImage (but not implemented)
            self.process"""
        super().__init__(name)
        self.game = game
        
        if tileSeperation == 'Auto':
            self.tileSep = self.tileSize/3.75
        else:
            self.tileSep = tileSeperation
        
        self.tileFull = self.tileSize+self.tileSep
        self.back = background
        
        self.imageUpdate = True
    
    def getRect(self):
        """Return a Rect object representing this row's area."""
        wh = self.wh[0]-self.tileSep*2, self.wh[1]-self.tileSep*2
        location = self.location[0]-wh[0]/2, self.location[1]-wh[1]/2
        return Rect(location, wh)
    
    def clearImage(self, tileDimentions):
        """Reset self.image using tileDimentions tuple and fills with self.back. Also updates self.wh."""
        tw, th = tileDimentions
        self.wh = Vector2(round(tw*self.tileFull+self.tileSep), round(th*self.tileFull+self.tileSep))
        self.image = getTileContainerImage(self.wh, self.back)
    
    def renderTile(self, tileObj, tileLoc):
        """Blit the surface of a given tile object onto self.image at given tile location. It is assumed that all tile locations are xy tuples."""
        x, y = tileLoc
        surf = getTileImage(tileObj, self.tileSize, self.greyshift)
        self.image.blit(surf, (round(x*self.tileFull+self.tileSep), round(y*self.tileFull+self.tileSep)))
    
    def updateImage(self):
        """Called when processing image changes, directed by self.imageUpdate being True."""
        raise NotImplemented
    
    def process(self, time_passed):
        """Call self.updateImage() if self.imageUpdate is True, then set self.updateImage to False."""
        if self.imageUpdate:
            self.updateImage()
            self.imageUpdate = False
    
    def get_data(self):
        """Return the data that makes this TileRenderer special."""
        data = super().get_data()
        data['tsp'] = f'{math.floor(self.tileSep*10):x}'
        data['tfl'] = f'{math.floor(self.tileFull*10):x}'
        if self.back is None:
            data['bac'] = 'N'
        else:
            data['bac'] = ''.join((f'{i:02x}' for i in self.back))
        return data
    
    def from_data(self, data):
        """Update this TileRenderer from data."""
        super().from_data(data)
        self.tileSep = int(f"0x{data['tsp']}", 16)/10
        self.tileFull = int(f"0x{data['tfl']}", 16)/10
        if data['bac'] == 'N':
            self.back = None
        else:
            lst = [int(f"0x{data['bac'][i:i+1]}", 16) for i in range(0, 6, 2)]
            self.back = tuple(lst)
    pass

class Cursor(TileRenderer):
    """Cursor Object."""
    greyshift = GREYSHIFT
    RenderPriority = 'last'
    def __init__(self, game):
        """Initialize cursor with a game it belongs to."""
        TileRenderer.__init__(self, 'Cursor', game, 'Auto', None)
        
        self.holdingNumberOne = False
        self.tiles = deque()
    
    def updateImage(self):
        """Update self.image."""
        self.clearImage((len(self.tiles), 1))
        
        for x in range(len(self.tiles)):
            self.renderTile(self.tiles[x], (x, 0))
    
    def isPressed(self):
        """Return True if the right mouse button is pressed."""
        return bool(pygame.mouse.get_pressed()[0])
    
    def getHeldCount(self, countNumberOne=False):
        """Return the number of held tiles, can be discounting number one tile."""
        l = len(self.tiles)
        if self.holdingNumberOne and not countNumberOne:
            return l-1
        return l
    
    def isHolding(self, countNumberOne=False):
        """Return True if the mouse is dragging something."""
        return self.getHeldCount(countNumberOne) > 0
    
    def getHeldInfo(self, includeNumberOne=False):
        """Returns color of tiles are and number of tiles held."""
        if not self.isHolding(includeNumberOne):
            return None, 0
        return self.tiles[0], self.getHeldCount(includeNumberOne)
    
    def process(self, time_passed):
        """Process cursor."""
        x, y = pygame.mouse.get_pos()
        x = saturate(x, 0, SCREENSIZE[0])
        y = saturate(y, 0, SCREENSIZE[1])
        self.location = (x, y)
        if self.imageUpdate:
            if len(self.tiles):
                self.updateImage()
            else:
                self.image = None
            self.imageUpdate = False
    
    def forceHold(self, tiles):
        """Pretty much it's drag but with no constraints."""
        for tile in tiles:
            if tile.color == NUMBERONETILE:
                self.holdingNumberOne = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.imageUpdate = True
    
    def drag(self, tiles):
        """Drag one or more tiles, as long as it's a list."""
        for tile in tiles:
            if not tile is None and tile.color == NUMBERONETILE:
                self.holdingNumberOne = True
                self.tiles.append(tile)
            else:
                self.tiles.appendleft(tile)
        self.imageUpdate = True
    
    def drop(self, number='All', allowOneTile=False):
        """Return all of the tiles the Cursor is carrying"""
        if self.isHolding(allowOneTile):
            if number == 'All':
                number = self.getHeldCount(allowOneTile)
            else:
                number = saturate(number, 0, self.getHeldCount(allowOneTile))
            
            tiles = []
            for tile in (self.tiles.popleft() for i in range(number)):
                if tile.color == NUMBERONETILE:
                    if not allowOneTile:
                        self.tiles.append(tile)
                        continue
                tiles.append(tile)
            self.imageUpdate = True
            
            self.holdingNumberOne = NUMBERONETILE in {tile.color for tile in self.tiles}
            return tiles
        return []
    
    def dropOneTile(self):
        """If holding the number one tile, drop it (returns it)."""
        if self.holdingNumberOne:
            notOne = self.drop('All', False)
            one = self.drop(1, True)
            self.drag(notOne)
            self.holdingNumberOne = False
            return one[0]
        return None
    
    def get_data(self):
        """Return all the data that makes this object special."""
        data = super().get_data()
        tiles = [t.get_data() for t in self.tiles]
        data['Ts'] = tiles
        return data
    
    def from_data(self, data):
        """Update this Cursor object from data."""
        super().from_data(data)
        self.tiles.clear()
        self.drag([Tile.from_data(t) for t in self.tiles])
    pass

def gscBoundIndex(boundsFalureReturn=None):
    """Return a decorator for any grid or grid subclass that will keep index positions within bounds."""
    def gscBoundsKeeper(function, boundsFalureReturnValue=None):
        """Grid or Grid Subclass Decorator that keeps index positions within bounds, as long as index is first argument after self arg."""
        @wraps(function)
        def keepWithinBounds(self, index, *args, **kwargs):
            """Wraper function that makes sure a position tuple (x, y) is valid."""
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
    def __init__(self, size, game, tileSeperation='Auto', background=TILEDEFAULT):
        """Grid Objects require a size and game at least."""
        TileRenderer.__init__(self, 'Grid', game, tileSeperation, background)
        
        self.size = tuple(size)
        
        self.data = array([Tile(-6) for i in range(int(self.size[0]*self.size[1]))]).reshape(self.size)
    
    def updateImage(self):
        """Update self.image."""
        self.clearImage(self.size)
        
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                self.renderTile(self.data[x, y], (x, y))
    
    def getTilePoint(self, screenLocation):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        # Can't get tile if screen location doesn't intersect our hitbox!
        if not self.pointIntersects(screenLocation):
            return None
        # Otherwise, find out where screen point is in image locations
        bx, by = self.toImageSurfLoc(screenLocation)#board x and y
        # Finally, return the full divides (no decimals) of xy location by self.tileFull.
        return int(bx // self.tileFull), int(by // self.tileFull)
    
    @gscBoundIndex()
    def placeTile(self, xy, tile):
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
        """Return a Tile Object from a given position in the grid if permitted. Return None on falure."""
        x, y = xy
        tileCopy = self.data[x, y]
        if tileCopy.color < 0:
            return None
        self.data[x, y] = Tile(replace)
        self.imageUpdate = True
        return tileCopy
    
    @gscBoundIndex()
    def getInfo(self, xy):
        """Return the Tile Object at a given position without deleteing it from the Grid."""
        x, y = xy
        return self.data[x, y]
    
    def getColors(self):
        """Return a list of the colors of tiles within self."""
        colors = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                infoTile = self.getInfo((x, y))
                if not infoTile.color in colors:
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
        data['w'] = int(self.size[0])
        data['h'] = int(self.size[1])   
        tiles = [f'{self.getInfo((x, y)).color+7:x}' for x in range(self.size[0]) for y in range(self.size[1])]
        data['Ts'] = ''.join(tiles)
        return data
    
    def from_data(self, data):
        """Update data in this board object."""
        super().from_data(data)
        self.size = int(data['w']), int(data['h'])
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                c = data['Ts'][x+y]
                self.data[x, y] = Tile(int(f'0x{c}', 16)-7)
        self.updateImage = True
    
    def __del__(self):
        super().__del__()
        del self.data
    pass

class Board(Grid):
    """Represents the board in the Game."""
    size = (5, 5)
    bcolor = ORANGE
    def __init__(self, player, variant_play=False):
        """Requires a player object."""
        Grid.__init__(self, self.size, player.game, background=self.bcolor)
        self.name = 'Board'
        self.player = player
        
        self.variant_play = variant_play
        self.additions = {}
        
        self.wallTiling = False
    
    def __repr__(self):
        return 'Board(%r, %s)' % (self.player, self.variant_play)
    
    def setColors(self, keepReal=True):
        """Reset tile colors."""
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                if not keepReal or self.data[x, y].color < 0:
                    self.data[x, y].color = -((self.size[1]-y+x)%REGTILECOUNT+1)
##                print(self.data[x, y].color, end=' ')
##            print()
##        print('-'*10)
    
    def getRow(self, index):
        """Return a row from self. Does not delete data from internal grid."""
        return [self.getInfo((x, index)) for x in range(self.size[0])]
    
    def getColumn(self, index):
        """Return a column from self. Does not delete data from internal grid."""
        return [self.getInfo((index, y)) for y in range(self.size[1])]
    
    def getColorsInRow(self, index, excludeNegs=True):
        """Return the colors placed in a given row in internal grid."""
        rowColors = [tile.color for tile in self.getRow(index)]
        if excludeNegs:
            rowColors = [c for c in rowColors if c >= 0]
        ccolors = Counter(rowColors)
        return list(sorted(ccolors.keys()))
    
    def getColorsInColumn(self, index, excludeNegs=True):
        """Return the colors placed in a given row in internal grid."""
        columnColors = [tile.color for tile in self.getColumn(index)]
        if excludeNegs:
            columnColors = [c for c in columnColors if c >= 0]
        ccolors = Counter(columnColors)
        return list(sorted(ccolors.keys()))
    
    def isWallTiling(self):
        """Return True if in Wall Tiling Mode."""
        return self.wallTiling
    
    def getTileForCursorByRow(self, row):
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
        return not tile.color in colors
    
    def getRowsToTile(self):
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
                    floor = self.player.getObjectByName('FloorLine')
                    floor.placeTile(self.additions[row])
                    self.additions[row] = None
    
    @gscBoundIndex(False)
    def wallTileFromPoint(self, position):
        """Given a position, wall tile. Return success on placement. Also updates if in wall tiling mode."""
        success = False
        column, row = position
        atPoint = self.getInfo(position)
        if atPoint.color <= 0:
            if row in self.additions:
                tile = self.additions[row]
                if not tile is None:
                    if self.canPlaceTileColorAtPoint(position, tile):
                        self.placeTile(position, tile)
                        self.additions[row] = column
                        # Update invalid placements after new placement
                        self.removeInvalidAdditions()
                        success = True
        if not self.getRowsToTile():
            self.wallTiling = False
        return success
    
    def wallTilingMode(self, movedDict):
        """Set self into Wall Tiling Mode. Finishes automatically if not in varient play mode."""
        self.wallTiling = True
        for key, value in ((key, movedDict[key]) for key in movedDict.keys()):
            key = int(key)-1
            if key in self.additions:
                raise RuntimeError('Key %r Already in additions dictionary!' % key)
            self.additions[key] = value
        if not self.variant_play:
            for row in range(self.size[1]):
                if row in self.additions:
                    rowdata = [tile.color for tile in self.getRow(row)]
                    tile = self.additions[row]
                    if tile is None:
                        continue
                    negTileColor = -(tile.color+1)
                    if negTileColor in rowdata:
                        column = rowdata.index(negTileColor)
                        self.placeTile((column, row), tile)
                        # Set data to the column placed in, use for scoring
                        self.additions[row] = column
                    else:
                        raise RuntimeError('%i not in row %i!' % (negTileColor, row))
                else:
                    raise RuntimeError(f'{row} not in movedDict!')
            self.wallTiling = False
        else:
            # Invalid additions can only happen in variant play mode.
            self.removeInvalidAdditions()
        pass
    
    @gscBoundIndex(([], []))
    def getTouchesContinuous(self, xy):
        """Return two lists, each of which contain all the tiles that touch the tile at given x y position, including that position."""
        rs, cs = self.size
        x, y = xy
        # Get row and column tile color data
        row = [tile.color for tile in self.getRow(y)]
        column = [tile.color for tile in self.getColumn(x)]
        # Both
        def gt(v, size, data):
            """Go through data foreward and backward from point v out by size, and return all points from data with a value >= 0."""
            def trng(rng, data):
                """Try range. Return all of data in range up to when indexed value is < 0."""
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
        # Return all of the self.getInfo points for each value in lst.
        getAll = lambda lst: [self.getInfo(pos) for pos in lst]
        # Get row touches
        rowTouches = comb(gt(x, rs, row), [y]*rs)
        # Get column touches
        columnTouches = comb([x]*cs, gt(y, cs, column))
        # Get real tiles from indexes and return
        return getAll(rowTouches), getAll(columnTouches)
    
    def scoreAdditions(self):
        """Using self.additions, which is set in self.wallTilingMode(), return the number of points the additions scored."""
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
    
    def getFilledRows(self):
        """Return the number of filled rows on this board."""
        count = 0
        for row in range(self.size[1]):
            real = (t.color >= 0 for t in self.getRow(row))
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
        tiles = (self.getInfo((x, y)) for x in range(self.size[0]) for y in range(self.size[1]))
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
        data['Wt'] = int(self.wallTiling)
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
        """Update this Board object from data."""
        super().from_data(data)
        self.wallTiling = bool(data['Wt'])
        for k in range(len(data['Ad'])):
            rv = data['Ad'][k]
            if rv == 'n':
                v = None
            elif rv.isupper():
                v = Tile(ord(rv)-65-6)
            else:
                v = int(rv)
            self.additions[k] = v
    pass

class Row(TileRenderer):
    """Represents one of the five rows each player has."""
    greyshift = GREYSHIFT
    def __init__(self, player, size, tilesep='Auto', background=None):
        TileRenderer.__init__(self, 'Row', player.game, tilesep, background)
        self.player = player
        self.size = int(size)
        
        self.color = -6
        self.tiles = deque([Tile(self.color)]*self.size)
    
    def __repr__(self):
        return 'Row(%r, %i, ...)' % (self.game, self.size)
    
    @classmethod
    def from_list(cls, player, iterable):
        """Return a new Row Object from a given player and an iterable of tiles."""
        lst = deque(iterable)
        obj = cls(player, len(lst))
        obj.color = None
        obj.tiles = lst
        return obj
    
    def updateImage(self):
        """Update self.image."""
        self.clearImage((self.size, 1))
        
        for x in range(len(self.tiles)):
            self.renderTile(self.tiles[x], (x, 0))
    
    def getTilePoint(self, screenLocation):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        xy = Grid.getTilePoint(self, screenLocation)
        if xy is None:
            return None
        x, y = xy
        return self.size-1-x
    
    def getPlaced(self):
        """Return the number of tiles in self that are not fake tiles, like grey ones."""
        return len([tile for tile in self.tiles if tile.color >= 0])
    
    def getPlaceable(self):
        """Return the number of tiles permitted to be placed on self."""
        return self.size - self.getPlaced()
    
    def isFull(self):
        """Return True if this row is full."""
        return self.getPlaced() == self.size
    
    def getInfo(self, location):
        """Return tile at location without deleteing it. Return None on invalid location."""
        index = self.size-1-location
        if index < 0 or index > len(self.tiles):
            return None
        return self.tiles[index]
    
    def canPlace(self, tile):
        """Return True if permitted to place given tile object on self."""
        placeable = (tile.color == self.color) or (self.color < 0 and tile.color >= 0)
        colorCorrect = tile.color >= 0 and tile.color < 5
        numCorrect = self.getPlaceable() > 0
        
        board = self.player.getObjectByName('Board')
        colorNotPresent = not tile.color in board.getColorsInRow(self.size-1)
        
        return placeable and colorCorrect and numCorrect and colorNotPresent
    
    def getTile(self, replace=-6):
        """Return the leftmost tile while deleteing it from self."""
        self.tiles.appendleft(Tile(replace))
        self.imageUpdate = True
        return self.tiles.pop()
    
    def placeTile(self, tile):
        """Place a given Tile Object on self if permitted."""
        if self.canPlace(tile):
            self.color = tile.color
            self.tiles.append(tile)
            end = self.tiles.popleft()
            if not end.color < 0:
                raise RuntimeError('Attempted deleteion of real tile from Row!')
            self.imageUpdate = True
    
    def canPlaceTiles(self, tiles):
        """Return True if permitted to place all of given tiles objects on self."""
        if len(tiles) > self.getPlaceable():
            return False
        for tile in tiles:
            if not self.canPlace(tile):
                return False
        tileColors = []
        for tile in tiles:
            if not tile.color in tileColors:
                tileColors.append(tile.color)
        if len(tileColors) > 1:
            return False
        return True
    
    def placeTiles(self, tiles):
        """Place multiple tile objects on self if permitted."""
        if self.canPlaceTiles(tiles):
            for tile in tiles:
                self.placeTile(tile)
    
    def wallTile(self, addToDict, blankColor=-6):
        """Move tiles around and into add dictionary for the wall tiling phase of the game. Removes tiles from self."""
        if not 'toBox' in addToDict:
            addToDict['toBox'] = []
        if not self.isFull():
            addToDict[str(self.size)] = None
            return
        else:
            self.color = blankColor
        addToDict[str(self.size)] = self.getTile()
        for i in range(self.size-1):
            addToDict['toBox'].append(self.getTile())
    
    def setBackground(self, color):
        """Set the background color for this row."""
        self.back = color
        self.imageUpdate = True
    
    def get_data(self):
        """Return the data that makes this Row special."""
        data = super().get_data()
        data['c'] = hex(self.color+7)[2:]
        data['s'] = str(self.size)
        data['Ts'] = ''.join([f'{t.color+7:x}' for t in self.tiles])
        return data
    
    def from_data(self, data):
        """Update this Row from data."""
        super().from_data(data)
        self.color = int(f"0x{data['c']}", 16)-7
        self.size = int(data['s'])
        self.tiles.clear()
        for i in range(len(data['Ts'])):
            c = data['Ts'][i]
            self.tiles.append(Tile(int(f'0x{c}', 16)-7))
    pass 

class PatternLine(MultipartObject):
    """Represents multiple rows to make the pattern line."""
    size = (5, 5)
    def __init__(self, player, rowSeperation=0):
        MultipartObject.__init__(self, 'PatternLine')
        self.player = player
        self.rowSep = rowSeperation
        
        for x, y in zip(range(self.size[0]), range(self.size[1])):
            self.add_object(Row(self.player, x+1))
        
        self.setBackground(None)
        
        self._lastloc = 0, 0
    
    def setBackground(self, color):
        """Set the background color for all rows in the pattern line."""
        self.set_attr_all('back', color)
        self.set_attr_all('imageUpdate', True)
    
    def getRow(self, row):
        """Return given row."""
        return self.get_object(row)
    
    def resetPosition(self):
        """Reset Locations of Rows according to self.location."""
        last = self.size[1]
        w = self.getRow(last-1).wh[0]
        if w is None:
            raise RuntimeError('Image Dimentions for Row Object (row.wh) are None!')
        h1 = self.getRow(0).tileFull
        h = last*h1
        self.wh = w, h
        w1 = h1/2
        
        x, y = self.location
        y -= h/2-w1
        for rid in self.objects:
            l = last-self.objects[rid].size
            self.objects[rid].location = x+(l*w1), y+rid*h1
    
    def getTilePoint(self, screenLocation):
        """Return the xy choordinates of which tile intersects given a point. Returns None if no intersections."""
        for y in range(self.size[1]):
            x = self.getRow(y).getTilePoint(screenLocation)
            if not x is None:
                return x, y
    
    def isFull(self):
        """Return True if self is full."""
        for rid in range(self.size[1]):
            if not self.getRow(rid).isFull():
                return False
        return True
    
    def wallTiling(self):
        """Return a dictionary to be used with wall tiling. Removes tiles from rows."""
        values = {}
        for rid in range(self.size[1]):
            self.getRow(rid).wallTile(values)
        return values
    
    def process(self, time_passed):
        """Process all the rows that make up the pattern line."""
        if self.hidden != self._lasthidden:
            self.set_attr_all('imageUpdate', True)
        super().process(time_passed)
    pass

class Text(Object):
    """Text object, used to render text with a given font."""
    def __init__(self, fontSize, color, background=None, cx=True, cy=True, name=''):
        Object.__init__(self, f'Text{name}')
        self.font = Font(FONT, fontSize, color, cx, cy, True, background, True)
        self._cxy = cx, cy
        self._last = None
    
    def getImageZero(self):
        """Return the screen location of the topleft point of self.image."""
        x = self.location[0]
        y = self.location[1]
        if self._cxy[0]:
            x -= self.wh[0]/2
        if self._cxy[1]:
            y -= self.wh[1]/2
        return x, y 
    
    def __repr__(self):
        return '<Text Object>'
    
    @staticmethod
    def get_font_height(font, size):
        """Return the height of font at fontsize size."""
        return pygame.font.Font(font, size).get_height()
    
    def updateValue(self, text, size=None, color=None, background='set'):
        """Return a surface of given text rendered in FONT."""
        if background == 'set':
            self.image = self.font.render_nosurf(text, size, color)
            return self.image
        self.image = self.font.render_nosurf(text, size, color, background)
        return self.image
    
    def getSurf(self):
        """Return self.image."""
        return self.image
    
    def getTilePoint(self, location):
        """Set getTilePoint attribute so that errors are not raised."""
        return None
    
    def process(self, time_passed):
        """Process text."""
        if self.font.lastText != self._last:
            self.updateValue(self.font.lastText)
            self._last = self.font.lastText
    
    def get_data(self):
        """Return the data that makes this Text Object special."""
        data = super().get_data()
        data['faa'] = int(self.font.antialias)
        gethex = lambda itera: ''.join((f'{i:02x}' for i in itera))
        if self.font.background is None:
            data['bg'] = 'N'
        else:
            data['bg'] = gethex(self.font.background)
        data['fc'] = gethex(self.font.color)
        data['fdc'] = int(self.font.doCache)
        data['flt'] = self.font.lastText
        return data
    
    def from_data(self, data):
        """Update this Text Object from data."""
        super().from_data(data)
        self.font.antialias = bool(data['faa'])
        getcolor = lambda itera: tuple([int(f"0x{itera[i:i+1]}", 16) for i in range(0, 6, 2)])
        if data['bac'] == 'N':
            self.font.background = None
        else:
            self.font.background = getcolor(data['bac'])
        self.font.color = getcolor(data['fc'])
        self.font.doCache = bool(data['fdc'])
        self.font.lastText = data['flt']
    pass

class FloorLine(Row):
    """Represents a player's floor line."""
    size = 7
    numberOneColor = NUMBERONETILE
    def __init__(self, player):
        Row.__init__(self, player, self.size, background=ORANGE)
        self.name = 'FloorLine'
        
##        self.font = Font(FONT, round(self.tileSize*1.2), color=BLACK, cx=False, cy=False)
        self.text = Text(round(self.tileSize*1.2), BLACK, cx=False, cy=False)
        self.hasNumberOne = False
        
        gen = floorLineSubGen(1)
        self.numbers = [next(gen) for i in range(self.size)]
    
    def __repr__(self):
        return 'FloorLine(%r)' % self.player
    
    def render(self, surface):
        """Update self.image."""
        Row.render(self, surface)
        
        sx, sy = self.location
        if self.wh is None:
            return
        w, h = self.wh
        for x in range(self.size):
            xy = round(x*self.tileFull+self.tileSep+sx-w/2), round(self.tileSep+sy-h/2)
            self.text.updateValue(str(self.numbers[x]))
            self.text.location = xy
            self.text.render(surface)
##            self.font.render(surface, str(self.numbers[x]), xy)
    
    def placeTile(self, tile):
        """Place a given Tile Object on self if permitted."""
        self.tiles.insert(self.getPlaced(), tile)
        
        if tile.color == self.numberOneColor:
            self.hasNumberOne = True
        
        boxLid = self.player.game.getObjectByName('BoxLid')
        
        def handleEnd(end):
            """Handle the end tile we are replacing. Ensures number one tile is not removed."""
            if not end.color < 0:
                if end.color == self.numberOneColor:
                    handleEnd(self.tiles.pop())
                    self.tiles.appendleft(end)
                    return
                boxLid.addTile(end)
        
        handleEnd(self.tiles.pop())
        
        self.imageUpdate = True
    
    def scoreTiles(self):
        """Score self.tiles and return how to change points."""
        runningTotal = 0
        for x in range(self.size):
            if self.tiles[x].color >= 0:
                runningTotal += self.numbers[x]
            elif x < self.size-1:
                if self.tiles[x+1].color >= 0:
                    raise RuntimeError('Player is likely cheating! Invalid placement of FloorLine tiles!')
        return runningTotal
    
    def getTiles(self, emtpyColor=-6):
        """Return tuple of tiles gathered, and then either the number one tile or None."""
        tiles = []
        numberOneTile = None
        for tile in (self.tiles.pop() for i in range(len(self.tiles))):
            if tile.color == self.numberOneColor:
                numberOneTile = tile
                self.hasNumberOne = False
            elif tile.color >= 0:
                tiles.append(tile)
        
        for i in range(self.size):
            self.tiles.append(Tile(emtpyColor))
        self.imageUpdate = True
        return tiles, numberOneTile
    
    def canPlaceTiles(self, tiles):
        """Return True."""
        return True
    
    def get_data(self):
        """Return the data that makes this FloorLine Row special."""
        data = super().get_data()
        data['fnt'] = self.font.get_data()
        return data
    
    def from_data(self, data):
        """Updata this FloorLine from data."""
        super().from_data(data)
        self.font.from_data(data['fnt'])
    pass

class Factory(Grid):
    """Represents a Factory."""
    size = (2, 2)
    color = WHITE
    outline = BLUE
    outSize = 0.1
    def __init__(self, game, factoryId):
        Grid.__init__(self, self.size, game, background=None)
        self.number = factoryId
        self.name = f'Factory{self.number}'
        
        self.radius = math.ceil(self.tileFull * self.size[0] * self.size[1] / 3 + 3)
    
    def __repr__(self):
        return 'Factory(%r, %i)' % (self.game, self.number)
    
    def addCircle(self, surface):
        if not f'FactoryCircle{self.radius}' in self.game.cache:
            rad = math.ceil(self.radius)
            surf = setAlpha(pygame.surface.Surface((2*rad, 2*rad)), 1)
            pygame.draw.circle(surf, self.outline, (rad, rad), rad)
            pygame.draw.circle(surf, self.color, (rad, rad), math.ceil(rad*(1-self.outSize)))
            self.game.cache[f'FactoryCircle{self.radius}'] = surf
        surf = self.game.cache[f'FactoryCircle{self.radius}'].copy()
        surface.blit(surf, (round(self.location[0]-self.radius), round(self.location[1]-self.radius)))
    
    def render(self, surface):
        """Render Factory."""
        if not self.hidden:
            self.addCircle(surface)
        super().render(surface)
    
    def fill(self, tiles):
        """Fill self with tiles. Will raise exception if insufficiant tiles."""
        if len(tiles) < self.size[0] * self.size[1]:
            raise RuntimeError('Insufficiant quantity of tiles! Needs %i!' % self.size[0] * self.size[1])
        for y in range(self.size[1]):
            for tile, x in zip((tiles.pop() for i in range(self.size[0])), range(self.size[0])):
                self.placeTile((x, y), tile)
        if tiles:
            raise RuntimeError('Too many tiles!')
    
    def grab(self):
        """Return all tiles on this factory."""
        return [tile for tile in (self.getTile((x, y)) for x in range(self.size[0]) for y in range(self.size[1])) if tile.color != -6]
    
    def grabColor(self, color):
        """Return all tiles of color given in the first list, and all non-matches in the seccond list."""
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
        data['n'] = self.number
        data['r'] = f'{math.ceil(self.radius):x}'
        return data
    
    def from_data(self, data):
        """Update this Factory from data."""
        super().from_data(data)
        self.number = int(data['n'])
        self.name = f'Factory{self.number}'
        self.radius = int(f"0x{data['r']}", 16)
    pass

class Factories(MultipartObject):
    """Factories Multipart Object, made of multiple Factory Objects."""
    teach = 4
    def __init__(self, game, factories:int, size='Auto'):
        """Requires a number of factories."""
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
                
        self.divyUpTiles()
    
    def __repr__(self):
        return 'Factories(%r, %i, ...)' % (self.game, self.count)
    
    def resetPosition(self):
        """Reset the position of all factories within."""
        degrees = 360 / self.count
        for i in range(self.count):
            rot = math.radians(degrees * i)
            self.objects[i].location = math.sin(rot)*self.size + self.location[0], math.cos(rot)*self.size + self.location[1]
    
    def process(self, time_passed):
        """Process factories. Does not react to cursor if hidden."""
        super().process(time_passed)
        if not self.hidden:
            cursor = self.game.getObjectByName('Cursor')
            if cursor.isPressed() and not cursor.isHolding():
                obj, point = self.getWhereTouches(cursor.location)
                if not obj is None and not point is None:
                    oid = int(obj[7:])
                    tileAtPoint = self.objects[oid].getInfo(point)
                    if (not tileAtPoint is None) and tileAtPoint.color >= 0:
                        table = self.game.getObjectByName('TableCenter')
                        select, tocenter = self.objects[oid].grabColor(tileAtPoint.color)
                        if tocenter:
                            table.add_tiles(tocenter)
                        cursor.drag(select)
    
    def divyUpTiles(self, emptyColor=-6):
        """Divy up tiles to each factory from the bag."""
        # For every factory we have,
        for fid in range(self.count):
            # Draw tiles for the factory
            drawn = []
            for i in range(self.teach):
                # If the bag is not empty,
                if not self.game.bag.isEmpty():
                    # Draw a tile from the bag.
                    drawn.append(self.game.bag.draw_tile())
                else:# Otherwise, get the box lid
                    boxLid = self.game.getObjectByName('BoxLid')
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
    
    def isAllEmpty(self):
        """Return True if all factories are empty."""
        for fid in range(self.count):
            if not self.objects[fid].isEmpty():
                return False
        return True
    
    def get_data(self):
        """Return what makes this Factories ObjectHandler special."""
        data = super().get_data()
        data['cnt'] = f'{self.count:x}'
        data['sz'] = f'{math.ceil(self.size):x}'
        return data
    
    def from_data(self, data):
        """Update these Factories with data."""
        super().from_data(data)
        self.count = int(f"0x{data['cnt']}", 16)
        self.size = int(f"0x{data['sz']}", 16)
    pass

class TableCenter(Grid):
    """Object that represents the center of the table."""
    size = (6, 6)
    firstTileColor = NUMBERONETILE
    def __init__(self, game, hasOne=True):
        """Requires a game object handler to exist in."""
        Grid.__init__(self, self.size, game, background=None)
        self.game = game
        self.name = 'TableCenter'
        
        self.firstTileExists = False
        if hasOne:
            self.add_number_one_tile()
        
        self.nextPosition = (0, 0)
    
    def __repr__(self):
        return 'TableCenter(%r)' % self.game
    
    def add_number_one_tile(self):
        """Add the number one tile to the internal grid."""
        if not self.firstTileExists:
            x, y = self.size
            self.placeTile((x-1, y-1), Tile(self.firstTileColor))
            self.firstTileExists = True
    
    def add_tile(self, tile):
        """Add a Tile Object to the Table Center Grid."""
        self.placeTile(self.nextPosition, tile)
        x, y = self.nextPosition
        x += 1
        y += int(x // self.size[0])
        x %= self.size[0]
        y %= self.size[1]
        self.nextPosition = (x, y)
        self.imageUpdate = True
    
    def add_tiles(self, tiles, sort=True):
        """Add multiple Tile Objects to the Table Center Grid."""
        yes = []
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
                    if self.getInfo((x, y)).color == self.firstTileColor:
                        continue
                at = self.getTile((x, y), replace)
                
                if not at is None:
                    full.append(at)
        sortedTiles = sorted(full, key=sortTiles)
        self.nextPosition = (0, 0)
        self.add_tiles(sortedTiles, False)
    
    def pull_tiles(self, tileColor, replace=-6):
        """Remove all of the tiles of tileColor from the Table Center Grid."""
        toPull = []
        for y in range(self.size[1]):
            for x in range(self.size[0]):
                infoTile = self.getInfo((x, y))
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
            cursor = self.game.getObjectByName('Cursor')
            if cursor.isPressed() and not cursor.isHolding() and not self.isEmpty():
                if self.pointIntersects(cursor.location):
                    point = self.getTilePoint(cursor.location)
                    # Shouldn't return none anymore since we have pointIntersects now.
                    colorAtPoint = self.getInfo(point).color
                    if colorAtPoint >= 0 and colorAtPoint < 5:
                        cursor.drag(self.pull_tiles(colorAtPoint))
        super().process(time_passed)
    
    def get_data(self):
        """Return what makes the TableCenter special."""
        data = super().get_data()
        data['fte'] = int(self.firstTileExists)
        x, y = self.nextPosition
        data['np'] = f'{x}{y}'
        return data
    
    def from_data(self, data):
        """Update the TableCenter from data."""
        super().from_data(data)
        self.firstTileExists = bool(data['fte'])
        x, y = data['np']
        self.nextPosition = int(x), int(y)
    pass

class Bag(object):
    """Represents the bag full of tiles."""
    def __init__(self, numTiles=100, tileTypes=5):
        self.numTiles = int(numTiles)
        self.tileTypes = int(tileTypes)
        self.tileNames = [chr(65+i) for i in range(self.tileTypes)]
        self.percentEach = (self.numTiles/self.tileTypes)/100
        self.full_reset()
    
    def full_reset(self):
        """Reset the bag to a full, re-randomized bag."""
        self.tiles = gen_random_proper_seq(self.numTiles, **{tileName:self.percentEach for tileName in self.tileNames})
    
    def __repr__(self):
        return 'Bag(%i, %i)' % (self.numTiles, self.tileTypes)
    
    def reset(self):
        """Randomize all the tiles in the bag."""
        self.tiles = deque(randomize(self.tiles))
    
    def get_color(self, tileName):
        """Return the color of a named tile."""
        if not tileName in self.tileNames:
            raise ValueError('Tile Name %s Not Found!' % tileName)
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
            raise ValueError('Invalid Tile Color!')
    
    def add_tile(self, tileObject):
        """Add a Tile Object to the bag."""
        name = self.get_name(int(tileObject.color))
        rnge = (0, len(self.tiles)-1)
        if rnge[1]-rnge[0] <= 1:
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
    pass

class BoxLid(Object):
    """BoxLid Object, represents the box lid were tiles go before being added to the bag again."""
    def __init__(self, game):
        Object.__init__(self, 'BoxLid')
        self.game = game
        self.tiles = deque()
    
    def __repr__(self):
        return 'BoxLid(%r)' % self.game
    
    def addTile(self, tile):
        """Add a tile to self."""
        if tile.color >= 0 and tile.color < 5:
            self.tiles.append(tile)
        else:
            raise Warning(f'BoxLid.addTile tried to add an invalid tile to self ({tile.color}). Be careful, bad things might be trying to happen.')
    
    def addTiles(self, tiles):
        """Add multiple tiles to self."""
        for tile in tiles:
            self.addTile(tile)
    
    def getTiles(self):
        """Return all tiles in self while deleteing them from self."""
        return [self.tiles.popleft() for i in range(len(self.tiles))]
    
    def isEmpty(self):
        """Return True if self is empty (no tiles on it)."""
        return len(self.tiles) == 0
    
    def get_data(self):
        """Return what makes this BoxLid Object special."""
        data = super().get_data()
        data['Ts'] = ''.join((f"{t.color+7:x}" for t in self.tiles))
        return data
    
    def from_data(self, data):
        """Update this BoxLid from data."""
        super().from_data(data)
        self.tiles.clear()
        self.addTiles((Tile(int(f"0x{t}", 16)-7) for t in data['Ts']))
    pass

class Player(MultipartObject):
    """Repesents a player. Made of lots of objects."""
    def __init__(self, game, playerId:int, networked=False, varient_play=False):
        """Requires a player Id and can be told to be controled by the network or be in varient play mode."""
        MultipartObject.__init__(self, 'Player%i' % playerId)
        
        self.game = game
        self.pid = playerId
        self.networked = networked
        self.varient_play = varient_play
                
        self.add_object(Board(self, self.varient_play))
        self.add_object(PatternLine(self))
        self.add_object(FloorLine(self))
        self.add_object(Text(SCOREFONTSIZE, SCORECOLOR))
        
        self.score = 0
        self.isTurn = False
        self.isWallTiling = False
        self.justHeld = False
        self.justDropped = False
        
        self.updateScore()
        
        self._lastloc = 0, 0
    
    def __repr__(self):
        return 'Player(%r, %i, %s, %s)' % (self.game, self.pid, self.networked, self.varient_play)
    
    def updateScore(self):
        """Update the scorebox for this player."""
        scoreBox = self.getObjectByName('Text')
        scoreBox.updateValue(f'Player {self.pid+1}: {self.score}')
    
    def turnNow(self):
        """It is this player's turn now."""
        if not self.isTurn:
            patternLine = self.getObjectByName('PatternLine')
            if self.isWallTiling:
                board = self.getObjectByName('Board')
                rows = board.getRowsToTile()
                for rowpos in rows:
                    patternLine.getRow(rowpos).setBackground(getTileColor(rows[rowpos], board.greyshift))
            else:
                patternLine.setBackground(PATSELECTCOLOR)
        self.isTurn = True
    
    def endOfTurn(self):
        """It is no longer this player's turn."""
        if self.isTurn:
            patternLine = self.getObjectByName('PatternLine')
            patternLine.setBackground(None)
        self.isTurn = False
    
    def itsTheEnd(self):
        """Function called by end state when game is over; Hide pattern lines and floor line."""
        pattern = self.getObjectByName('PatternLine')
        floor = self.getObjectByName('FloorLine')
        
        pattern.hidden = True
        floor.hidden = True
    
    def resetPosition(self):
        """Reset positions of all parts of self based off self.location."""
        x, y = self.location
        
        bw, bh = self.getObjectByName('Board').wh
        self.getObjectByName('Board').location = x+bw/2, y
        lw = self.getObjectByName('PatternLine').wh[0]/2
        self.getObjectByName('PatternLine').location = x-lw, y
        fw = self.getObjectByName('FloorLine').wh[0]
        self.getObjectByName('FloorLine').location = x-lw*(2/3)+TILESIZE/3.75, y+bh*(2/3)
        self.getObjectByName('Text').location = x-(bw/3), y-(bh*(2/3))
    
    def wallTiling(self):
        """Do the wall tiling phase of the game for this player."""
        self.isWallTiling = True
        patternLine = self.getObjectByName('PatternLine')
        floorLine = self.getObjectByName('FloorLine')
        board = self.getObjectByName('Board')
        boxLid = self.game.getObjectByName('BoxLid')
        
        data = patternLine.wallTiling()
        boxLid.addTiles(data['toBox'])
        del data['toBox']
        
        board.wallTilingMode(data)
    
    def doneWallTiling(self):
        """Return True if internal Board is done wall tiling."""
        board = self.getObjectByName('Board')
        return not board.isWallTiling()
    
    def nextRound(self):
        """Called when player is done wall tiling."""
        self.isWallTiling = False
    
    def scorePhase(self):
        """Do the scoring phase of the game for this player."""
        board = self.getObjectByName('Board')
        floorLine = self.getObjectByName('FloorLine')
        boxLid = self.game.getObjectByName('BoxLid')
        def saturatescore():
            if self.score < 0:
                self.score = 0
        
        self.score += board.scoreAdditions()
        self.score += floorLine.scoreTiles()
        saturatescore()
        
        toBox, numberOne = floorLine.getTiles()
        boxLid.addTiles(toBox)
        
        self.updateScore()
        
        return numberOne
    
    def endOfGameScoring(self):
        """Update final score with additional end of game points."""
        board = self.getObjectByName('Board')
        
        self.score += board.endOfGameScoreing()
        
        self.updateScore()
    
    def hasHorizLine(self):
        """Return True if this player has a horizontal line on their game board filled."""
        board = self.getObjectByName('Board')
        return board.hasFilledRow()
    
    def getHorizontalLines(self):
        """Return the number of filled horizontal lines this player has on their game board."""
        board = self.getObjectByName('Board')
        return board.getFilledRows()
    
    def process(self, time_passed):
        """Process Player."""
        if self.isTurn:# Is our turn?
            if self.hidden and self.isWallTiling and self.varient_play:
                # If hidden, not anymore. Our turn.
                self.hidden = False
            if not self.networked:# We not networked.
                cursor = self.game.getObjectByName('Cursor')
                boxLid = self.game.getObjectByName('BoxLid')
                patternLine = self.getObjectByName('PatternLine')
                floorLine = self.getObjectByName('FloorLine')
                board = self.getObjectByName('Board')
                if cursor.isPressed():# Mouse down?
                    obj, point = self.getWhereTouches(cursor.location)
                    if not obj is None and not point is None:# Something pressed
                        if cursor.isHolding():# Cursor holding tiles
                            madeMove = False
                            if not self.isWallTiling:# Is wall tiling:
                                if obj == 'PatternLine':
                                    pos, rowNum = point
                                    row = patternLine.getRow(rowNum)
                                    if not row.isFull():
                                        info = row.getInfo(pos)
                                        if not info is None and info.color < 0:
                                            color, held = cursor.getHeldInfo()
                                            todrop = min(pos+1, row.getPlaceable())
                                            tiles = cursor.drop(todrop)
                                            if row.canPlaceTiles(tiles):
                                                row.placeTiles(tiles)
                                                madeMove = True
                                            else:
                                                cursor.forceHold(tiles)
                                elif obj == 'FloorLine':
                                    tilesToAdd = cursor.drop()
                                    if floorLine.isFull():# Floor is full,
                                        # Add tiles to box instead.
                                        boxLid.addTiles(tilesToAdd)
                                    elif floorLine.getPlaceable() < len(tilesToAdd):
                                        # Add tiles to floor line and then to box
                                        while len(tilesToAdd) > 0:
                                            if floorLine.getPlaceable() > 0:
                                                floorLine.placeTile(tilesToAdd.pop())
                                            else:
                                                boxLid.addTile(tilesToAdd.pop())
                                    else:# Otherwise add to floor line for all.
                                        floorLine.placeTiles(tilesToAdd)
                                    madeMove = True
                            elif not self.justHeld: # Cursor holding and wall tiling
                                if obj == 'Board':
                                    atPoint = board.getInfo(point)
                                    if atPoint.color == -6:
                                        column, row = point
                                        cursorTile = cursor.drop(1)[0]
                                        boardTile = board.getTileForCursorByRow(row)
                                        if not boardTile is None:
                                            if cursorTile.color == boardTile.color:
                                                if board.wallTileFromPoint(point):
                                                    self.justDropped = True
                                                    patternLine.getRow(row).setBackground(None)
                            
                            if madeMove:
                                if not self.isWallTiling:
                                    if cursor.holdingNumberOne:
                                        floorLine.placeTile(cursor.dropOneTile())
                                    if cursor.getHeldCount(True) == 0:
                                        self.game.nextTurn()
                        else:# Mouse down, something pressed, and not holding anything
                            if self.isWallTiling:# Wall tiling, pressed, not holding
                                if obj == 'Board':
                                    if not self.justDropped:
                                        columnNum, rowNum = point
                                        tile = board.getTileForCursorByRow(rowNum)
                                        if not tile is None:
                                            cursor.drag([tile])
                                            self.justHeld = True
                else: # Mouse up
                    if self.justHeld:
                        self.justHeld = False
                    if self.justDropped:
                        self.justDropped = False
            if self.isWallTiling and self.doneWallTiling():
                self.nextRound()
                self.game.nextTurn()
        self.set_attr_all('hidden', self.hidden)
        super().process(time_passed)
    
    def get_data(self):
        """Return what makes this Player MultipartObject special."""
        data = super().get_data()
        data['pi'] = int(self.pid)
        data['sc'] = f'{self.score:x}'
        data['tu'] = int(self.isTurn)
        data['iwt'] = int(self.isWallTiling)
        return data
    
    def from_data(self, data):
        """Update this Player from data."""
        super().from_data()
        self.pid = int(data['pi'])
        self.score = int(data['sc'], 16)
        self.isTurn = bool(data['tu'])
        self.isWallTiling = bool(data['iwt'])
    pass

class Button(Text):
    """Button Object."""
    textcolor = BUTTONTEXTCOLOR
    backcolor = BUTTONBACKCOLOR
    def __init__(self, state, name, minSize=10, initValue='', fontSize=BUTTONFONTSIZE):
        super().__init__(fontSize, self.textcolor, background=None)
        self.name = f'Button{name}'
        self.state = state
        
        self.minsize = int(minSize)
        self.updateValue(initValue)
        
        self.borderWidth = math.floor(fontSize/12)#5
        
        self.action = lambda: None
        self.delay = 0.6
        self.curTime = 1
    
    def __repr__(self):
        return f'Button({self.name[6:]}, {self.state}, {self.minsize}, {self.font.lastText}, {self.font.pyfont})'
    
    def get_height(self):
        return self.font.get_height()
    
    def bind_action(self, function):
        """When self is pressed, call given function exactly once. Function takes no arguments."""
        self.action = function
    
    def updateValue(self, text, size=None, color=None, background='set'):
        disp = str(text).center(self.minsize)
        super().updateValue(f' {disp} ', size, color, background)
        self.font.lastText = disp
    
    def render(self, surface):
        if not self.hidden:
            text_rect = self.getRect()
            if PYGAME_VERSION < 201:
                pygame.draw.rect(surface, self.backcolor, text_rect)
                pygame.draw.rect(surface, BLACK, text_rect, self.borderWidth)
            else:
                pygame.draw.rect(surface, self.backcolor, text_rect, border_radius=20)
                pygame.draw.rect(surface, BLACK, text_rect, width=self.borderWidth, border_radius=20)
        super().render(surface)
    
    def isPressed(self):
        """Return True if this button is pressed."""
        cursor = self.state.game.getObjectByName('Cursor')
        if not self.hidden and cursor.isPressed():
            if self.pointIntersects(cursor.location):
                return True
        return False
    
    def process(self, time_passed):
        """Call self.action one time when pressed, then wait self.delay to call again."""
        if self.curTime > 0:
            self.curTime = max(self.curTime-time_passed, 0)
        else:
            if self.isPressed():
                self.action()
                self.curTime = self.delay
        if self.font.lastText != self._last:
            self.textSize = self.font.pyfont.size(f' {self.font.lastText} ')
        super().process(time_passed)
    
    def from_data(self, data):
        """Update this Button from data."""
        super().from_data(data)
        self.updateValue(data['flt'])
    pass

class GameState(object):
    """Base class for all game states."""
    name = 'Base Class'
    def __init__(self, name):
        """Initialize state with a name, set self.game to None to be overwritten later."""
        self.game = None
        self.name = name
    
    def __repr__(self):
        return f'<GameState {self.name}>'
    
    def entry_actions(self):
        """Preform entry actions for this GameState."""
        pass
    
    def do_actions(self):
        """Preform actions for this GameState."""
        pass
    
    def check_state(self):
        """Check state and return new state. None remains in current state."""
        return None
    
    def exit_actions(self):
        """Preform exit actions for this GameState."""
        pass
    pass

class MenuState(GameState):
    """Game State where there is a menu with buttons."""
    buttonMin = 10
    fontsize = BUTTONFONTSIZE
    def __init__(self, name):
        """Initialize GameState and set up self.bh."""
        super().__init__(name)
        self.bh = Text.get_font_height(FONT, self.fontsize)
        
        self.toState = None
    
    def addButton(self, name, value, action, location='Center', size=fontsize, minlen=buttonMin):
        """Add a new Button object to self.game with arguments. Return button id."""
        button = Button(self, name, minlen, value, size)
        button.bind_action(action)
        if location != 'Center':
            button.location = location
        self.game.add_object(button)
        return button.id
    
    def addText(self, name, value, location, color=BUTTONTEXTCOLOR, cx=True, cy=True, size=fontsize):
        """Add a new Text object to self.game with arguments. Return text id."""
        text = Text(size, color, None, cx, cy, name)
        text.location = location
        text.updateValue(value)
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
            f"""Set MenuState.toState to {stateName}."""
            self.toState = stateName
        return toStateName
    
    def var_dependant_to_state(self, **kwargs):
        """attribute name = (target value, on trigger tostate)."""
        for state in kwargs:
            if not len(kwargs[state]) == 2:
                raise ValueError(f'Key "{state}" is invalid!')
            key, value = kwargs[state]
            if not hasattr(self, key):
                raise ValueError(f'{self} object does not have attribute "{key}"!')
        def toStateByAttributes():
            """Set MenuState.toState to a new state if conditions are right."""
            for state in kwargs:
                key, value = kwargs[state]
                if getattr(self, key) == value:
                    self.toState = state
        return toStateByAttributes
    
    def with_update(self, updateFunction):
        """Return a wrapper for a function that will call updateFunction after function."""
        def update_wrapper(function):
            """Wrapper for any function that could require a screen update."""
            @wraps(function)
            def function_with_update():
                """Call main function, then update function."""
                function()
                updateFunction()
            return function_with_update
        return update_wrapper
    
    def updateText(self, textName, valueFunc):
        """Update text object with textName's display value."""
        def updater():
            f"""Update text object {textName}'s value with {valueFunc}."""
            text = self.game.getObjectByName(f'Text{textName}')
            text.updateValue(valueFunc())
        return updater
    
    def toggleButtonState(self, textname, boolattr, textfunc):
        """Return function that will toggle the value of text object <textname>, toggleing attribute <boolattr>, and setting text value with textfunc."""
        def valfunc():
            """Return the new value for the text object. Gets called AFTER value is toggled."""
            return textfunc(getattr(self, boolattr))
        @self.with_update(self.updateText(textname, valfunc))
        def toggleValue():
            """Toggle the value of boolattr."""
            self.set_var(boolattr, not getattr(self, boolattr))
        return toggleValue
    
    def check_state(self):
        """Return self.toState."""
        return self.toState
    pass

class InitState(GameState):
    def __init__(self):
        super().__init__('Init')
    
    def entry_actions(self):
        self.game.keyboard.addListener('\x7f', 'Delete')
        self.game.keyboard.bindAction('Delete', 'screenshot', 5)
        
        self.game.keyboard.addListener('\x1b', 'Escape')
        self.game.keyboard.bindAction('Escape', 'raiseClose', 5)
        
        self.game.keyboard.addListener('0', 'Debug')
        self.game.keyboard.bindAction('Debug', 'debug', 5)
    
    def check_state(self):
        return 'Title'
    pass

class TitleScreen(MenuState):
    """Game state when the title screen is up."""
    def __init__(self):
        super().__init__('Title')
    
    def entry_actions(self):
        super().entry_actions()
        sw, sh = SCREENSIZE
        self.addButton('ToSettings', 'New Game', self.to_state('Settings'), (sw/2, sh/2-self.bh*0.5))
        self.addButton('ToCredits', 'Credits', self.to_state('Credits'), (sw/2, sh/2+self.bh*3), self.fontsize/1.5)
        self.addButton('Quit', 'Quit', self.game.raiseClose, (sw/2, sh/2+self.bh*4), self.fontsize/1.5)
    pass

class CreditsScreen(MenuState):
    """Game state when credits for original game are up."""
    def __init__(self):
        super().__init__('Credits')
    
    def entry_actions(self):
        super().entry_actions()
    
    def check_state(self):
        return 'Title'
    pass

class SettingsScreen(MenuState):
    """Game state when user is defining game type, players, etc."""
    def __init__(self):
        super().__init__('Settings')
        
        self.playerCount = 0#2
        self.hostMode = True
        self.variant_play = False
    
    def entry_actions(self):
        """Add cursor object and tons of button and text objects to the game."""
        super().entry_actions()
        
        def addNumbers(start, end, widthEach, cx, cy):
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
                
                @self.with_update(self.updateText('Players', lambda: f'Players: {self.playerCount}'))
                def setPlayerCount():
                    f"""Set varibable playerCount to {display} while updating text."""
                    return self.set_var('playerCount', display)
                
                self.addButton(f'SetCount{number}', str(display), setPlayerCount,
                               (cx+(widthEach*x), cy), size=self.fontsize/1.5, minlen=3)
            for i in range(count):
                addNumber(i, start+i, cx, cy)
        
        sw, sh = SCREENSIZE
        cx = sw/2
        cy = sh/2
        
        hostText = lambda x: f'Host Mode: {x}'
        self.addText('Host', hostText(self.hostMode), (cx, cy-self.bh*3))
        self.addButton('ToggleHost', 'Toggle', self.toggleButtonState('Host', 'hostMode', hostText), (cx, cy-self.bh*2), size=self.fontsize/1.5)
        
        # TEMPORARY: Hide everything to do with "Host Mode", networked games arn't done yet.
        self.game.set_attr_all('hidden', True)
        
        varientText = lambda x: f'Varient Play: {x}'
        self.addText('Varient', varientText(self.variant_play), (cx, cy-self.bh))
        self.addButton('ToggleVarient', 'Toggle', self.toggleButtonState('Varient', 'variant_play', varientText), (cx, cy), size=self.fontsize/1.5)
        
        self.addText('Players', f'Players: {self.playerCount}', (cx, cy+self.bh))
        addNumbers(2, 4, 70, cx, cy+self.bh*2)
        
        varToState = self.var_dependant_to_state(FactoryOffer=('hostMode', True), FactoryOfferNetworked=('hostMode', False))
        self.addButton('StartGame', 'Start Game', varToState, (cx, cy+self.bh*3))
    
    def exit_actions(self):
        self.game.start_game(self.playerCount, self.variant_play, self.hostMode)
        self.game.bag.full_reset()
    pass

class PhaseFactoryOffer(GameState):
    """Game state when it's the Factory Offer Stage."""
    def __init__(self):
        super().__init__('FactoryOffer')
    
    def entry_actions(self):
        """Advance turn."""
        self.game.nextTurn()
    
    def check_state(self):
        """If all tiles are gone, go to wall tiling. Otherwise keep waiting for that to happen."""
        fact = self.game.getObjectByName('Factories')
        table = self.game.getObjectByName('TableCenter')
        cursor = self.game.getObjectByName('Cursor')
        if fact.isAllEmpty() and table.isEmpty() and not cursor.isHolding(True):
            return 'WallTiling'
        return None
    pass

class PhaseFactoryOfferNetworked(PhaseFactoryOffer):
    def __init__(self):
        GameState.__init__(self, 'FactoryOfferNetworked')
    
    def check_state(self):
        return 'WallTilingNetworked'
    pass

class PhaseWallTiling(GameState):
    def __init__(self):
        super().__init__('WallTiling')
    
    def entry_actions(self):
        self.nextStarter = None
        self.notProcessed = []
        
        self.game.playerTurnOver()
        
        # For each player,
        for pid in range(self.game.players):
            # Activate wall tiling mode.
            player = self.game.getPlayer(pid)
            player.wallTiling()
            # Add that player's pid to the list of not-processed players.
            self.notProcessed.append(player.pid)
        
        # Start processing players.
        self.game.nextTurn()
    
    def do_actions(self):
        if self.notProcessed:
            if self.game.playerTurn in self.notProcessed:
                player = self.game.getPlayer(self.game.playerTurn)
                if player.doneWallTiling():
                    # Once player is done wall tiling, score their moves.
                    numberOne = player.scorePhase()#Also gets if they had the number one tile.
                    if numberOne:
                        # If player had the number one tile, remember that.
                        self.nextStarter = self.game.playerTurn
                        # Then, add the number one tile back to the table center.
                        table = self.game.getObjectByName('TableCenter')
                        table.add_number_one_tile()
                    # After calculating their score, delete player from un-processed list
                    self.notProcessed.remove(self.game.playerTurn)
                    # and continue to the next un-processed player.
                    self.game.nextTurn()
            else:
                self.game.nextTurn()
    
    def check_state(self):
        cursor = self.game.getObjectByName('Cursor')
        if not self.notProcessed and not cursor.isHolding():
            return 'PrepareNext'
        return None
    
    def exit_actions(self):
        # Set up the player that had the number one tile to be the starting player next round.
        self.game.playerTurnOver()
        # Goal: make (self.playerTurn + 1) % self.players = self.nextStarter
        nturn = self.nextStarter - 1
        if nturn < 0:
            nturn += self.game.players
        self.game.playerTurn = nturn
    pass

class PhaseWallTilingNetworked(PhaseWallTiling):
    def __init__(self):
        GameState.__init__(self, 'WallTilingNetworked')
    
    def check_state(self):
        return 'PrepareNextNetworked'
    pass

class PhasePrepareNext(GameState):
    def __init__(self):
        super().__init__('PrepareNext')
    
    def entry_actions(self):
        players = (self.game.getPlayer(pid) for pid in range(self.game.players))
        complete = (player.hasHorizLine() for player in players)
        self.newRound = not any(complete)
    
    def do_actions(self):
        if self.newRound:
            fact = self.game.getObjectByName('Factories')
            # This also handles bag re-filling from box lid.
            fact.divyUpTiles()
    
    def check_state(self):
        if self.newRound:
            return 'FactoryOffer'
        return 'End'
    pass

class PhasePrepareNextNetworked(PhasePrepareNext):
    def __init__(self):
        GameState.__init__(self, 'PrepareNextNetworked')
    
    def check_state(self):
        return 'EndNetworked'
    pass

class EndScreen(MenuState):
    def __init__(self):
        super().__init__('End')
        self.ranking = {}
        self.wininf = ''
    
    def get_winners(self):
        """Update self.ranking by player scores."""
        self.ranking = {}
        scpid = {}
        for pid in range(self.game.players):
            player = self.game.getPlayer(pid)
            player.itsTheEnd()
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
                players = [self.game.getPlayer(pid) for pid in pids]
                lines = [(p.getHorizontalLines(), p.pid) for p in players]
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
        table = self.game.getObjectByName('TableCenter')
        table.hidden = True
        
        fact = self.game.getObjectByName('Factories')
        fact.set_attr_all('hidden', True)
        
        # Add buttons
        bid = self.addButton('ReturnTitle', 'Return to Title', self.to_state('Title'), (SCREENSIZE[0]/2, math.floor(SCREENSIZE[1]*(4/5))))
        buttontitle = self.game.get_object(bid)
        buttontitle.RenderPriority = 'last-1'
        buttontitle.curTime = 2
        
        # Add score board
        x = SCREENSIZE[0]/2
        y = 10
        idx = 0
        for line in self.wininf.split('\n'):
            lid = self.addText(f'Line{idx}', line, (x, y), cx=True, cy=False)
##            self.game.get_object(bid).RenderPriority = f'last{-(2+idx)}'
            self.game.get_object(bid).RenderPriority = f'last-2'
            idx += 1
            y += self.bh
        pass
    pass

class EndScreenNetworked(EndScreen):
    def __init__(self):
        MenuState.__init__(self, 'EndNetworked')
        self.ranking = {}
        self.wininf = ''
    
    def check_state(self):
        return 'Title'
    pass

class Game(ObjectHandler):
    """Game object, contains most of what's required for Azul."""
    tileSize = 30
    def __init__(self):
        ObjectHandler.__init__(self)
        self.keyboard = None#Gets overwritten by Keyboard object
        
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
                         PhaseFactoryOfferNetworked(),
                         PhaseWallTilingNetworked(),
                         PhasePrepareNextNetworked(),
                         EndScreenNetworked()])
        self.initializedState = False
        
        self.backgroundColor = BACKGROUND
        
        self.isHost = True
        self.players = 0
        self.factories = 0
        
        self.playerTurn = 0
        
        # Tiles
        self.bag = Bag(TILECOUNT, REGTILECOUNT)
        
        # Cache
        self.cache = {}
    
    def __repr__(self):
        return 'Game()'
    
    def isPressed(self, key):
        """Function that is ment to be overwritten by the Keyboard object."""
        return False
    
    def debug(self):
        """Debug."""
        pass
    
    def screenshot(self):
        """Save a screenshot of this game's most recent frame."""
        surface = pygame.surface.Surface(SCREENSIZE)
        self.render(surface)
        strTime = '-'.join(time.asctime().split(' '))
        filename = f'Screenshot_at_{strTime}.png'
                
        if not os.path.exists('Screenshots'):
            os.mkdir('Screenshots')
        
        surface.unlock()
        pygame.image.save(surface, os.path.join('Screenshots', filename),
                          filename)
        del surface
        
        savepath = os.path.join(os.getcwd(), 'Screenshots')
        
        print(f'Saved screenshot as "{filename}" in "{savepath}".')
    
    def raiseClose(self):
        """Raise a window close event."""
        pygame.event.post(pygame.event.Event(QUIT))
    
    def add_states(self, states):
        """Add game states to self."""
        for state in states:
            if not isinstance(state, GameState):
                raise ValueError(f'"{state}" Object is not a subclass of GameState!')
            state.game = self
            self.states[state.name] = state
    
    def set_state(self, new_state_name):
        """Change states and preform any exit / entry actions."""
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
    
    def updateState(self):
        """Preform the actions of the active state and potentially change states."""
        # Only continue if there is an active state
        if self.active_state is None:
            return
        
        # Preform the actions of the active state and check conditions
        self.active_state.do_actions()
        
        new_state_name = self.active_state.check_state()
        if not new_state_name is None:
            self.set_state(new_state_name)
    
    def add_object(self, obj):
        """Add an object to the game."""
        obj.game = self
        super().add_object(obj)
    
    def render(self, surface):
        "Render all of self.objects to the screen."""
        surface.fill(self.backgroundColor)
        self.renderObjects(surface)
    
    def process(self, time_passed):
        """Process all the objects and self."""
        if not self.initializedState and not self.keyboard is None:
            self.set_state('Init')
            self.initializedState = True
        self.processObjects(time_passed)
        self.updateState()
    
    def getPlayer(self, pid):
        """Get the player with player id pid."""
        if self.players:
            return self.getObjectByName(f'Player{pid}')
        raise RuntimeError('No players!')
    
    def playerTurnOver(self):
        """Call endOfTurn for current player."""
        if self.playerTurn >= 0 and self.playerTurn < self.players:
            oldPlayer = self.getPlayer(self.playerTurn)
            if oldPlayer.isTurn:
                oldPlayer.endOfTurn()
    
    def nextTurn(self):
        """Tell current player it's the end of their turn, and update who's turn it is and now it's their turn."""
        if self.isHost:
            self.playerTurnOver()
            last = self.playerTurn
            self.playerTurn = (self.playerTurn + 1) % self.players
            if self.playerTurn == last and self.players > 1:
                self.nextTurn()
                return
            newPlayer = self.getPlayer(self.playerTurn)
            newPlayer.turnNow()            
    
    def start_game(self, players, varient_play=False, hostMode=True, address=''):
        """Start a new game."""
        self.reset_cache()
        maxPlayers = 4
        self.players = saturate(players, 1, maxPlayers)
        self.isHost = hostMode
        self.factories = self.players * 2 + 1
        
        self.rm_star()
        
        self.add_object(Cursor(self))
        self.add_object(TableCenter(self))
        self.add_object(BoxLid(self))
        
        if self.isHost:
            self.bag.reset()
            self.playerTurn = random.randint(-1, self.players-1)
        else:
            self.playerTurn = 'Unknown'
        
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
        if self.isHost:
            self.nextTurn()
        
        factory = Factories(self, self.factories)
        factory.location = cx, cy
        self.add_object(factory)
        self.processObjects(0)
        
        if self.isHost:
            self.nextTurn()
    
    def screen_size_update(self):
        """Function called when screen size is updated."""
        objsWithAttr = self.get_objects_with_attr('screen_size_update')
        for oid in objsWithAttr:
            obj = self.get_object(oid)
            obj.screen_size_update()
    pass

class Keyboard(object):
    """Keyboard object, handles keyboard input."""
    def __init__(self, target, **kwargs):
        self.target = target
        self.target.keyboard = self
        self.target.isPressed = self.isPressed
        
        self.keys = {}      #Map of keyboard events to names
        self.actions = {}   #Map of keyboard event names to functions
        self.time = {}      #Map of names to time until function should be called again
        self.delay = {}     #Map of names to duration timer waits for function recalls
        self.active = {}    #Map of names to boolian of pressed or not
        
        if kwargs:
            for name in kwargs:
                if not hasattr(kwargs[name], '__iter__'):
                    raise ValueError('Keyword arguments must be given as name=[key, self.target.functionName, delay]')
                elif len(kwargs[name]) == 2:
                    key, functionName = kwargs[name]
                    delay = None
                elif len(kwargs[name]) == 3:
                    key, functionName, delay = kwargs[name]
                else:
                    raise ValueError
                self.addListener(key, name)
                self.bindAction(name, functionName)
    
    def __repr__(self):
        return 'Keyboard(%s)' % repr(self.target)
    
    def isPressed(self, key):
        """Return True if <key> is pressed."""
        if key in self.active:
            return self.active[key]
        return False
    
    def addListener(self, key:int, name:str):
        """Listen for key down events with event.key == key arguement and when that happens set self.actions[name] to true."""
        self.keys[key] = name#key to name
        self.actions[name] = lambda: None#name to function
        self.time[name] = 0#name to time until function recall
        self.delay[name] = None#name to function recall delay
        self.active[name] = False#name to boolian of pressed
    
    def getFunctionFromTarget(self, functionName:str):
        """Return function with name functionName from self.target"""
        if hasattr(self.target, functionName):
            return getattr(self.target, functionName)
        else:
            return lambda: None
    
    def bindAction(self, name:str, targetFunctionName:str, delay=None):
        """Bind an event we are listening for to calling a function, can call multiple times if delay is not None."""
        self.actions[name] = self.getFunctionFromTarget(targetFunctionName)
        self.delay[name] = delay
    
    def setActive(self, name:str, value:bool):
        """Set active value for key name <name> to <value>."""
        if name in self.active:
            self.active[name] = bool(value)
            if not value:
                self.time[name] = 0
    
    def setKey(self, key:int, value:bool, _nochar=False):
        """Set active value for key <key> to <value>"""
        if key in self.keys:
            self.setActive(self.keys[key], value)
        elif not _nochar:
            if key < 0x110000:
                self.setKey(chr(key), value, True)
    
    def readEvent(self, event):
        """Handles an event."""
        if event.type == KEYDOWN:
            self.setKey(event.key, True)
        elif event.type == KEYUP:
            self.setKey(event.key, False)
    
    def readEvents(self, events):
        """Handles a list of events."""
        for event in events:
            self.readEvent(event)
    
    def process(self, time_passed):
        """Sends commands to self.target based on pressed keys and time."""
        for name in self.active:
            if self.active[name]:
                self.time[name] = max(self.time[name] - time_passed, 0)
                if self.time[name] == 0:
                    self.actions[name]()
                    if not self.delay[name] is None:
                        self.time[name] = self.delay[name]
                    else:
                        self.time[name] = math.inf
    pass

def networkShutdown():
    try:
        pass
    except BaseException:
        pass

def run():
    global game
    global SCREENSIZE
    # Set up the screen
    screen = pygame.display.set_mode(SCREENSIZE, RESIZABLE, 16)
    pygame.display.set_caption(f'{__title__} {__version__}')
##    pygame.display.set_icon(pygame.image.load('icon.png'))
    pygame.display.set_icon(getTileImage(Tile(5), 32))
    
    # Set up the FPS clock
    clock = pygame.time.Clock()
    
    game = Game()
    keyboard = Keyboard(game)
    
    MUSIC_END = USEREVENT + 1#This event is sent when a music track ends
    
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
                # If it's not a quit or music end event, tell the keybord handler about it.
                keyboard.readEvent(event)
        
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
    "Save the last frame before the game crashed."
    surface = pygame.display.get_surface().copy()
    strTime = '-'.join(time.asctime().split(' '))
    filename = f'Crash_at_{strTime}.png'
    
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
        _, fails = pygame.init()
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
        networkShutdown()
