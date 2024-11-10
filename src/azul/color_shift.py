#!/usr/bin/env python3
# Color shift

"""Color shift"""

# Programmed by CoolCat467

__title__ = "Color shift"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from collections.abc import Iterable

from vector import Vector


def lerp(color: Iterable, to_color: Iterable, percent: float) -> tuple:
    """Linear interpolate from color to to_color by percent."""
    vcolor = Vector.from_iter(color)
    if len(vcolor) > len(to_color):
        to_color = tuple(to_color) + (0,) * (len(vcolor) - len(to_color))
    elif len(color) < len(to_color):
        to_color = to_color[: len(vcolor)]
    return tuple(round(vcolor.lerp(to_color, percent)))


def color_shift(surface, to_color: Iterable, percent: float):
    """Shift colors in pygame surface twards to_color by percent."""
    w, h = surface.get_size()
    surface.lock()
    for y in range(h):
        for x in range(w):
            color = surface.get_at((x, y))
            new = lerp(color, to_color, percent)
            surface.set_at((x, y), new)
    surface.unlock()
    return surface


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
##    run()
