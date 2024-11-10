#!/usr/bin/env python3
# Events module

"""Events module"""

# Programmed by CoolCat467

__title__ = "Events"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import asyncio
import sys as __sys
import traceback as __traceback
from collections import deque
from collections.abc import Generator
from io import StringIO
from typing import Any

from gears import BaseBot, Timer


def log_active_exception() -> str:
    """Log active exception"""
    # Get values from exc_info
    values = __sys.exc_info()
    # Get error message.
    msg = "#" * 16 + "\n"
    msg += "Exception class:\n" + str(values[0]) + "\n"
    msg += "Exception text:\n" + str(values[1]) + "\n"

    yes_totaly_a_file = StringIO()
    __traceback.print_exception(
        None,
        values[1],
        values[2],
        file=yes_totaly_a_file,
    )
    msg += "Traceback:\n" + yes_totaly_a_file.getvalue()[:-1] + "\n"
    msg += "#" * 16 + "\n"
    if values[0] is not None:
        print(msg)
    return msg


class Event:
    """Event"""

    __slots__: tuple = ("type", "args", "kwargs")

    def __init__(self, event_type: str, *args, **kwargs):
        self.type = event_type
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, index: int | str) -> Any:
        if isinstance(index, int):
            if index < len(self.args):
                return self.args[index]
        elif index in self.kwargs:
            return self.kwargs[index]
        raise ValueError("Index is not valid.")

    def __repr__(self) -> str:
        """Return event representation"""
        args = ", ".join(map(repr, self.args))
        kwargs = ", ".join(
            key + "=" + repr(value) for key, value in self.kwargs.items()
        )
        data = [repr(self.type)]
        if args:
            data.append(args)
        if kwargs:
            data.append(kwargs)
        return f'{self.__class__.__name__}({", ".join(data)})'


class ProcEvent(Event):
    """Processable Event"""

    __slots__: tuple = ("__processed",)

    def __init__(self, event_type: str, *args, **kwargs):
        super().__init__(event_type, *args, **kwargs)
        self.__processed = False

    @property
    def processed(self):
        """Bool of is this event has been Processed"""
        return self.__processed

    def process(self):
        """Mark this event as processed"""
        if self.__processed:
            raise RuntimeError("Event has already been processed!")
        self.__processed = True


class EventLoop(BaseBot):
    """Gear runner extension to have events"""

    __slots__ = ("events",)

    def __init__(self, eventloop):
        super().__init__(eventloop)

        self.events = deque()

    def submit_event(self, event: Event | ProcEvent) -> int:
        """Submit event to be processed. Return number of unprocessed events."""
        self.events.append(event)
        return len(self.events)

    def get_events(self) -> Generator:
        """Events generator. Yields submitted events."""
        while self.events:
            yield self.events.popleft()

    async def proc_additional_handlers(self, event: Event | ProcEvent) -> None:
        """Process additional handlers for an event. Called by process."""

    async def process(self) -> None:
        """Process all events for all gears if they are event handlers."""
        if not hasattr(self, "gears"):
            raise AttributeError(
                'No "gear" attribute defined! Need to subclass BaseBot too!',
            )
        handlers = []
        for name, gear in self.gears.items():
            if isinstance(gear, EventHandler):
                handlers.append(name)
        reprocess = []
        coros = []
        for event in self.get_events():
            for name in handlers:
                gear = self.get_gear(name)
                if gear is None:
                    continue
                coros.append(gear.read_event(event))
            coros.append(self.proc_additional_handlers(event))
            await asyncio.gather(*coros)
            coros.clear()

            if isinstance(event, ProcEvent):
                if not event.processed:
                    reprocess.append(event)
        for event in reprocess:
            self.submit_event(event)


class EventLoopProcessor(Timer):
    """EventLoopProcessor timer for in the case that runner is buisy elsewhere,"""

    min_delay: int = 1

    def __init__(self, runner: EventLoop):
        self.bot: EventLoop
        if not isinstance(runner, EventLoop):
            raise TypeError('"runner" argument is not an EventLoop subclass!')
        super().__init__(runner, "EventLoopProcessor", 1)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.running=}, {self.stopped=}>"

    async def tick(self) -> bool:
        """Process event loop. Return True on exception"""
        try:
            await self.bot.process()
        except Exception:
            log_active_exception()
            return True
        return False

    def on_stop(self) -> None:
        """Remove this gear"""
        ##        print(f'{self.__class__.__name__} stopped.')
        self.bot.remove_gear(self.name)


class EventHandler:
    """Gear extension to have events."""

    async def read_event(self, event: Event) -> None:
        """Read an event. Up to subclass to handle."""


def run() -> None:
    """Run example"""
    turtle = EventLoop(None)
    turtle.submit_event(Event("bob say hi", "hi", cat="best"))
    print(turtle.events)


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
    run()
