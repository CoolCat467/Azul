"""Crop functions - Functions to crop Surfaces."""

# Programmed by CoolCat467

from __future__ import annotations

# Copyright (C) 2020-2024  CoolCat467
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

__title__ = "Crop Functions"
__author__ = "CoolCat467"
__version__ = "0.0.0"


from typing import TYPE_CHECKING

from pygame.color import Color
from pygame.rect import Rect
from pygame.surface import Surface

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable


def crop_color(surface: Surface, color: Color) -> Surface:
    """Crop out color from surface."""
    w, h = surface.get_size()

    surf = surface.copy().convert_alpha()
    surf.fill(Color(0, 0, 0, 0))

    area = Rect(0, 0, 0, 0)
    surf.lock()
    y_inter = False
    x_inter = False
    for y in range(h):
        for x in range(w):
            value = surface.get_at((x, y))
            if value == color:
                if not y_inter:
                    area.top = y
                else:
                    area.bottom = y
                if not x_inter:
                    area.left = x
                else:
                    area.right = x
            else:
                if not y_inter:
                    y_inter = True
                if not x_inter:
                    x_inter = True
                surf.set_at((x, y), value)
    surf.unlock()
    final = Surface(area.size)
    final.blit(surf, (0, 0), area=area)
    return surf


def auto_crop_clear(
    surface: Surface,
    clear: Color | None = None,
) -> Surface:
    """Remove unneccicary pixels from image."""
    if clear is None:
        clear = Color(0, 0, 0, 0)
    surface = surface.convert_alpha()
    w, h = surface.get_size()
    surface.lock()

    def find_end(
        iterfunc: Callable[[int], Iterable[Color]],
        rangeobj: Iterable[int],
    ) -> int:
        for x in rangeobj:
            if not all(y == clear for y in iterfunc(x)):
                return x
        return x

    def column(x: int) -> Generator[Color, None, None]:
        return (surface.get_at((x, y)) for y in range(h))

    def row(y: int) -> Generator[Color, None, None]:
        return (surface.get_at((x, y)) for x in range(w))

    leftc = find_end(column, range(w))
    rightc = find_end(column, range(w - 1, -1, -1))
    topc = find_end(row, range(h))
    floorc = find_end(row, range(h - 1, -1, -1))
    surface.unlock()
    dim = Rect(leftc, topc, rightc - leftc, floorc - topc)
    return surface.subsurface(dim)


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.\n")
