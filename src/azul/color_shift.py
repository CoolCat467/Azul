"""Color shift."""

from __future__ import annotations

# Programmed by CoolCat467

__title__ = "Color shift"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from typing import TYPE_CHECKING

from azul.vector import Vector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pygame.surface import Surface


def lerp(
    color: Iterable[int],
    to_color: Iterable[int],
    percent: float,
) -> tuple[int, ...]:
    """Linear interpolate from color to to_color by percent."""
    vcolor = Vector.from_iter(color)
    if len(vcolor) > len(to_color):
        to_color = tuple(to_color) + (0,) * (len(vcolor) - len(to_color))
    elif len(color) < len(to_color):
        to_color = to_color[: len(vcolor)]
    return tuple(round(vcolor.lerp(to_color, percent)))


def color_shift(
    surface: Surface,
    to_color: Iterable[int],
    percent: float,
) -> Surface:
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
