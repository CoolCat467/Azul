#!/usr/bin/env python3
# TITLE DISCRIPTION
# -*- coding: utf-8 -*-

# Programmed by CoolCat467


import math
import random
from collections import deque

def lerp(a, b, i):
    """Linear enterpolate from A to B."""
    return a+(b-a)*i

def lerpColor(a, b, i):
    """Linear enterpolate from color a to color b."""
    r1, b1, g1 = a
    r2, b2, g2 = b
    data = lerp(r1, r2, i), lerp(b1, b2, i), lerp(g1, g2, i)
    return data

def saturate(value, low, high):
    """Keep value within min and max"""
    return min(max(value, low), high)

def randomize(iterable):
    """Randomize all values of an iterable."""
    lst = list(iterable)
    return [lst.pop(random.randint(0, len(lst)-1)) for i in range(len(lst))]

def gen_random_proper_seq(length, **kwargs):
    """Generates a random sequence of letters given keyword arguments of <letter>=<percentage in decimal>"""
    letters = []
    if sum(list(kwargs.values())) != 1:
        raise ArithmeticError('Sum of perentages of '+' '.join(list(kwargs.keys()))+' are not equal to 100 percent!')
    for letter in kwargs:
        letters += [letter] * math.ceil(length * kwargs[letter])
    return deque(randomize(letters))

def sortTiles(tileObj):
    """Function to be used when sorting tiles."""
    return tileObj.color

def getCacheSignatureTile(tile, tilesize, greyshift, outlineSize):
    """Return the string a tile and it's configuration information would use to identify itself in the tile cache."""
    safeFloat = lambda x: round(x*100)
    data = tile.color, safeFloat(tilesize), safeFloat(greyshift), safeFloat(outlineSize)
    return ''.join((str(i) for i in data))

def floorLineSubGen(seed=1):
    """Floor Line subtraction number generator. Can continue indefinitely."""
    num = seed
    while True:
        nxt = [-num] * (num+1)
        for i in nxt:
            yield i
        num += 1
