"""Tools."""

from __future__ import annotations

# Programmed by CoolCat467
from typing import TypeVar

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


# def getCacheSignatureTile(tile, tilesize, greyshift, outlineSize):
# """Return the string a tile and it's configuration information would use to identify itself in the tile cache."""
# safeFloat = lambda x: round(x*100)
# # types: error: Call to untyped function (unknown) in typed context
# data = tile.color, safeFloat(tilesize), safeFloat(greyshift), safeFloat(outlineSize)
# # types:           ^
# return ''.join((str(i) for i in data))
