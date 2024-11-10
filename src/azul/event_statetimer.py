#!/usr/bin/env python3
# EventStateTimer

"""EventStateTimer"""

# Programmed by CoolCat467

__title__ = "EventStateTimer"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from abc import ABC, abstractmethod

from events import Event, EventHandler, EventLoop
from gears import AsyncState, StateTimer


class EventAsyncState(AsyncState):
    """Async State that handles events"""

    __slots__: tuple = tuple()

    async def on_event(self, event):
        """Process an event"""


class StatorEventExtend(EventHandler, ABC):
    """Add submit_event and async read_event to a statemachine subclass"""

    @abstractmethod
    def submit_event(self, event: Event) -> None:
        """Submit an event to runner"""
        ...

    async def read_event(self, event):
        """Give event to statemachine activestate if it has a on_event handler."""
        # lintcheck: no-member (E1101): Instance of 'StatorEventExtend' has no 'active_state' member
        if hasattr(self.active_state, "on_event"):
            # lintcheck: no-member (E1101): Instance of 'StatorEventExtend' has no 'active_state' member
            await self.active_state.on_event(event)


class EventStateTimer(StateTimer, StatorEventExtend):
    """Timer Statemachine that is also an event handler."""

    def __init__(self, bot: EventLoop, name: str, delay: int = 0):
        self.bot: EventLoop
        StateTimer.__init__(self, bot, name, delay)
        StatorEventExtend.__init__(self)

    def submit_event(self, event):
        """Submit an event to runner"""
        return self.bot.submit_event(event)


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
