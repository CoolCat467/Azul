#!/usr/bin/env python3
# TITLE DISCRIPTION
# -*- coding: utf-8 -*-

# Programmed by CoolCat467

from typing import TypeVar, Generator, Any

import math
import random
from collections import deque

T = TypeVar('T')
I = TypeVar('I', int, float)

def lerp(a: I, b: I, i: float) -> float:
    """Linear enterpolate from A to B."""
    return a+(b-a)*i

def lerp_color(a: tuple[int, int, int],
              b: tuple[int, int, int],
              i: float) -> tuple[float, float, float]:
    """Linear enterpolate from color a to color b."""
    r1, b1, g1 = a
    r2, b2, g2 = b
    return lerp(r1, r2, i), lerp(b1, b2, i), lerp(g1, g2, i)


def saturate(value: I, low: I, high: I) -> I:
    """Keep value within min and max"""
    return min(max(value, low), high)

def randomize(iterable: list[T]) -> list[T]:
    """Randomize all values of an iterable."""
    lst = list(iterable)
    return [lst.pop(random.randint(0, len(lst)-1)) for i in range(len(lst))]

def gen_random_proper_seq(length: int, **kwargs: float) -> deque[str]:
    """Generates a random sequence of letters given keyword arguments of <letter>=<percentage in decimal>"""
    letters = []
    if sum(list(kwargs.values())) != 1:
        raise ArithmeticError('Sum of perentages of '+' '.join(list(kwargs.keys()))+' are not equal to 100 percent!')
    for letter in kwargs:
        letters += [letter] * math.ceil(length * kwargs[letter])
    return deque(randomize(letters))

def sortTiles(tileObj: Any) -> int:
    """Function to be used when sorting tiles."""
    return tileObj.color

##def getCacheSignatureTile(tile, tilesize, greyshift, outlineSize):
##    """Return the string a tile and it's configuration information would use to identify itself in the tile cache."""
##    safeFloat = lambda x: round(x*100)
##    # types: error: Call to untyped function (unknown) in typed context
##    data = tile.color, safeFloat(tilesize), safeFloat(greyshift), safeFloat(outlineSize)
##    # types:           ^
##    return ''.join((str(i) for i in data))

def floorLineSubGen(seed: int = 1) -> Generator[int, None, None]:
    """Floor Line subtraction number generator. Can continue indefinitely."""
    num = seed
    while True:
        nxt = [-num] * (num+1)
        for i in nxt:
            yield i
        num += 1
