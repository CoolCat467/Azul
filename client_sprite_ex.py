#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extended client sprites

"Extended client sprites"

# Programmed by CoolCat467

__title__ = 'Client sprite extensions'
__author__ = 'CoolCat467'
__version__ = '0.0.0'

from collections import deque

from vector import Vector
from client_sprite import ClientSprite

class MovingSprite(ClientSprite):
    "Moving sprite class"
    def __init__(self, *groups, **kwargs):
        super().__init__(*groups, **kwargs)

        self.destination = self.location

    async def on_event(self, event) -> None:
        "Process an event"
        if event.type == 'MouseButtonDown' and event['button'] == 1:
            self.destination = event['pos']

    def update(self, time_passed: float) -> None:
        "Update with time_passed"
        if self.location == self.destination:
            return
        heading = Vector.from_points(self.location, self.destination)
        distance_to_dest = heading.magnitude

        if distance_to_dest == 0:
            return

        speed = 200
        heading.normalize()
        travel_distance = min(distance_to_dest, (time_passed * speed))

        if travel_distance > 0:
            self.location += heading * travel_distance
            self.dirty = 1

class MoveListSprite(MovingSprite):
    "Moving in list sprite"
    def __init__(self, *groups, **kwargs):
        super().__init__(*groups, **kwargs)

        self.destinations = deque()

    async def on_event(self, event) -> None:
        "Add an event to destination que."
        if event.type == 'MouseButtonDown' and event['button'] == 1:
            self.destinations.append(event['pos'])

    def update(self, time_passed: float) -> None:
        if self.destinations:
            if self.location == self.destination:
                self.destination = self.destinations.popleft()
        super().update(time_passed)


def run():
    pass





if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.')
    run()
