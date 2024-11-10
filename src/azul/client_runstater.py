#!/usr/bin/env python3
# Runner

"""Not so crazy anymore! Ha HA!"""

# Programmed by CoolCat467

__title__ = "Runner"
__author__ = "CoolCat467"
__version__ = "0.0.0"


from event_statetimer import EventAsyncState
from events import EventLoop
from gears import AsyncState, AsyncStateMachine, StateTimerExitState


class Runner(AsyncStateMachine, EventLoop):
    """Asynchronous State Machine + Gear Runner.
    Important: In process, event proc AFTER think, THEN active state none check
    """

    __slots__ = ("running",)

    def __init__(self, eventloop):
        """Requires event loop to run gears in."""
        AsyncStateMachine.__init__(self)
        EventLoop.__init__(self, eventloop)

        self.active_state: EventAsyncState | AsyncState

        self.add_state(StateTimerExitState())

        self.running: bool = True

    @property
    def gear_close(self) -> bool:
        """Return True if not running."""
        return not self.running

    def submit_coro(self, coro):
        """Submit a coro as task for event loop to complete"""
        return self.loop.create_task(coro)

    async def proc_additional_handlers(self, event) -> None:
        """Process active state on_event handler if it exists."""
        ##        if hasattr(self.active_state, 'on_event'):
        if isinstance(self.active_state, EventAsyncState):
            await self.active_state.on_event(event)

    async def process(self) -> None:
        """Process gear events and state machine.
        Important: Think, THEN Event proc, THEN active state none check
        """
        await self.think()
        await super().process()

        if self.active_state is None:
            await self.close()

    async def close(self) -> None:
        await super().close()
        self.running = False


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
