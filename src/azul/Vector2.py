#!/usr/bin/env python3
# Vector2 Class for games
# -*- coding: utf-8 -*-

# Original version by Will McGugan, modified extensively by CoolCat467
# Programmed by CoolCat467

__title__ = "Vector2 Module"
__author__ = "CoolCat467"
__version__ = "1.1.0"
__ver_major__ = 1
__ver_minor__ = 1
__ver_patch__ = 0

import math
from collections.abc import Iterator
from typing import Union

Num = int | float
Point = Union["Vector2", tuple[Num, Num], list[Num]]


class Vector2:
    __slots__ = ("x", "y")
    """Vector2 Object. Takes an x an a y choordinate."""

    def __init__(
        self,
        x: list[Num] | tuple[Num, Num] | Num = 0,
        y: Num = 0,
    ) -> None:
        x_val: Num
        if isinstance(x, (list, tuple)):
            x_val, y = x
        else:
            x_val = x
        self.x: Num = x_val
        self.y: Num = y

    def __repr__(self) -> str:
        """Return representation of Vector2."""
        return f"Vector2({self.x}, {self.y})"

    @staticmethod
    def from_points(frompoint: Point, topoint: Point) -> "Vector2":
        """Return a vector with the direction of frompoint to topoint."""
        P1, P2 = list(frompoint), list(topoint)
        return Vector2(P2[0] - P1[0], P2[1] - P1[1])

    def get_magnitude(self) -> float:
        """Return the magnitude (length) of self."""
        return math.sqrt(self.x**2 + self.y**2)

    def get_distance_to(self, point: Point) -> float:
        """Return the magnitude (distance) to a given point."""
        return Vector2.from_points(point, self).get_magnitude()

    def normalize(self) -> None:
        """Normalize self (make into a unit vector) **IN PLACE**"""
        magnitude = self.get_magnitude()
        if magnitude != 0:
            self.x /= magnitude
            self.y /= magnitude

    def copy(self) -> "Vector2":
        """Return a copy of self."""
        return Vector2(self.x, self.y)

    def __copy__(self) -> "Vector2":
        """Return a copy of self."""
        return self.copy()

    def get_normalized(self) -> "Vector2":
        """Return a normalized vector (heading)."""
        vec = self.copy()
        vec.normalize()
        return vec

    def getHeading(self) -> float:
        """Returns the arc tangent (measured in radians) of self.y/self.x."""
        return math.atan2(self.y, self.x)

    def getHeadingDeg(self) -> float:
        """Returns the arc tangent (measured in degrees) of self.y/self.x."""
        return math.degrees(self.getHeading())

    def rotate(self, radians: float) -> "Vector2":
        """Returns a new vector by rotating self around (0, 0) by radians."""
        newHeading = self.getHeading() + radians
        magnitude = self.get_magnitude()
        x = math.cos(newHeading) * magnitude
        y = math.sin(newHeading) * magnitude
        # Round up to 13 digits, or we'll get weird rounding errors, like
        # .00000000000001
        return Vector2(round(x, 13), round(y, 13))

    def rotateDeg(self, degrees: float) -> "Vector2":
        """Returns a new vector by rotating self around (0, 0) by degrees clockwise."""
        return self.rotate(math.radians(-degrees))

    def _addv(self, vec: "Vector2") -> "Vector2":
        """Return the addition of self and another vector."""
        return Vector2(self.x + vec.x, self.y + vec.y)

    # rhs is Right Hand Side
    def __add__(self, rhs: Point) -> "Vector2":
        if isinstance(rhs, self.__class__):
            return self._addv(rhs)
        if hasattr(rhs, "__len__"):
            if len(rhs) == 2:
                x, y = rhs
                ##                return self._addv(self.__class__(x, y))
                return Vector2(self.x + x, self.y + y)
            raise LookupError(
                "Length of right hand sign operator length is not equal to two!",
            )
        raise AttributeError("Length not found.")

    def _subv(self, vec: "Vector2") -> "Vector2":
        """Return the subtraction of self and another vector."""
        print(self.x, self.y, vec.x, vec.y)
        return Vector2(self.x - vec.x, self.y - vec.y)

    def __sub__(self, rhs: Point) -> "Vector2":
        if isinstance(rhs, self.__class__):
            return self._subv(rhs)
        if hasattr(rhs, "__len__"):
            if len(rhs) == 2:
                x, y = rhs
                return Vector2(self.x - x, self.y - y)
            raise LookupError(
                "Length of right hand sign operator length is not equal to two!",
            )
        raise AttributeError("Length not found.")

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __mul__(self, scalar: Num) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: Num) -> "Vector2":
        try:
            x, y = self.x / scalar, self.y / scalar
        except ZeroDivisionError:
            x, y = self.x, self.y
        return Vector2(x, y)

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[Num]:
        return iter((self.x, self.y))

    def __getitem__(self, x: int) -> Num:
        return (self.x, self.y)[x]

    def __round__(self) -> "Vector2":
        return Vector2(round(self.x), round(self.y))

    def __abs__(self) -> "Vector2":
        return Vector2(abs(self.x), abs(self.y))
