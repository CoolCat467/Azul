"""Tools."""

from __future__ import annotations

# Programmed by CoolCat467
import math
import random
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from azul.game import Tile

T = TypeVar("T")
Numeric = TypeVar("Numeric", int, float)


def lerp(a: Numeric, b: Numeric, i: float) -> float:
    """Linear enterpolate from A to B."""
    return a + (b - a) * i


def lerp_color(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    i: float,
) -> tuple[float, float, float]:
    """Linear enterpolate from color a to color b."""
    r1, b1, g1 = a
    r2, b2, g2 = b
    return lerp(r1, r2, i), lerp(b1, b2, i), lerp(g1, g2, i)


def saturate(value: Numeric, low: Numeric, high: Numeric) -> Numeric:
    """Keep value within min and max."""
    return min(max(value, low), high)


def randomize(iterable: Iterable[T]) -> list[T]:
    """Randomize all values of an iterable."""
    lst = list(iterable)
    random.shuffle(lst)
    return lst


def gen_random_proper_seq(length: int, **kwargs: float) -> list[str]:
    """Generate a random sequence of letters given keyword arguments of <letter>=<percentage in decimal>."""
    letters = []
    if sum(list(kwargs.values())) != 1:
        raise ArithmeticError(
            "Sum of percentages of "
            + " ".join(list(kwargs.keys()))
            + " are not equal to 100 percent!",
        )
    for letter in kwargs:
        letters += [letter] * math.ceil(length * kwargs[letter])
    return randomize(letters)


def sort_tiles(tile_object: Tile) -> int:
    """Key function for sorting tiles."""
    return tile_object.color


##def getCacheSignatureTile(tile, tilesize, greyshift, outlineSize):
##    """Return the string a tile and it's configuration information would use to identify itself in the tile cache."""
##    safeFloat = lambda x: round(x*100)
##    # types: error: Call to untyped function (unknown) in typed context
##    data = tile.color, safeFloat(tilesize), safeFloat(greyshift), safeFloat(outlineSize)
##    # types:           ^
##    return ''.join((str(i) for i in data))


def floor_line_subtract_generator(seed: int = 1) -> Generator[int, None, None]:
    """Floor Line subtraction number generator. Can continue indefinitely."""
    num = seed
    while True:
        nxt = [-num] * (num + 1)
        yield from nxt
        num += 1
