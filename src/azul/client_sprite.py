#!/usr/bin/env python3
# Client Sprite and Renderer

"""Client Sprite and Renderer."""

# Programmed by CoolCat467

import asyncio
from collections.abc import Callable

from client_runstater import EventAsyncState, Runner
from events import Event
from pygame.event import event_name
from sprite import (
    GroupProcessor,
    LayeredDirty,
)

__title__ = "Client Sprite"
__author__ = "CoolCat467"
__version__ = "0.0.0"


class SubRenderer(LayeredDirty):
    """Gear Runner and Layered Dirty Sprite group."""

    __slots__: tuple = ()

    async def handle_event(self, event):
        """Process on_click handlers for sprites on mouse down events."""
        coros = []
        for sprite in self.sprites():
            if hasattr(sprite, "on_event"):
                coros.append(sprite.on_event(event))
        if event.type == "MouseButtonDown" and event["button"] == 1:
            sprites = self.get_sprites_at(event["pos"])
            layers = tuple(
                sorted({sprite.layer for sprite in sprites}, reverse=True),
            )
            for layered_sprite in sprites:
                if hasattr(layered_sprite, "on_click"):
                    coros.append(
                        layered_sprite.on_click(
                            layers.index(layered_sprite.layer),
                        ),
                    )
        await asyncio.gather(*coros)


class GroupGearProcessor(GroupProcessor, Runner):
    """Gear Runner and Layered Dirty Sprite group handler."""

    ##    __slots__ = ('config', 'groups')
    sub_renderer_class = SubRenderer

    def __init__(self, event_loop, config, language):
        Runner.__init__(self, event_loop)
        GroupProcessor.__init__(self)

        self.conf = config
        self.lang = language

    def send_pygame_event(self, py_event) -> None:
        """Send py event to event loop, adding pygame_event_prefix to name."""
        name = event_name(py_event.type)
        self.submit_event(Event(name, **py_event.dict))

    async def proc_additional_handlers(self, event: Event) -> None:
        """Process sprites with on_event handlers."""
        await super().proc_additional_handlers(event)
        coros = []
        for group in self.groups.values():
            coros.append(group.handle_event(event))

        await asyncio.gather(*coros)

    async def update(self, time_passed: float) -> None:
        """Process gears and update sprites."""
        super().update(time_passed)
        await self.process()


class RenderClientState(EventAsyncState):
    """Client state with a renderer."""

    __slots__ = ("group", "hault", "is_new_group")
    keep_newgroup = False

    def __init__(self, name: str):
        super().__init__(name)

        # typecheck: error: Incompatible types in assignment (expression has type "None", variable has type "int")
        self.group: int = None
        self.hault: bool = False
        self.is_new_group: bool = False

    @property
    def renderer(self):
        """Group render."""
        if self.group is None:
            self.is_new_group = True
            self.group = self.machine.new_group(self.name)
        return self.machine.get_group(self.group)

    @property
    def conf(self):
        """Configuration from client."""
        return self.machine.conf

    @property
    def lang(self):
        """Language from client."""
        return self.machine.lang

    async def on_event(self, event) -> None:
        if event.type == "UserEvent" and event["event"] == "escape":
            self.hault = True

    # typecheck: error: Missing return statement
    async def check_conditions(self) -> str | None:
        if self.hault:
            return "Hault"
        return None

    async def exit_actions(self) -> None:
        """Remove new group if not keep new groups."""
        self.hault = False
        if not self.keep_newgroup and self.is_new_group:
            self.renderer.empty()
            # typecheck: error: "AsyncStateMachine" has no attribute "remove_group"
            self.machine.remove_group(self.group)
            self.is_new_group = False
            # typecheck: error: Incompatible types in assignment (expression has type "None", variable has type "int")
            self.group = None


class MenuClientState(RenderClientState):
    """Menu state."""

    __slots__ = ("next_state",)

    def __init__(self, name: str):
        super().__init__(name)

        self.next_state = None

    def set_state(self, name: str) -> Callable:
        """Set state."""

        def set_state_wrapper() -> None:
            """Set next state."""
            # typecheck: error: Incompatible types in assignment (expression has type "str", variable has type "None")
            self.next_state = name

        return set_state_wrapper

    async def check_conditions(self) -> str | None:
        """Hault if self.hault, otherwise self.next_state."""
        if self.hault:
            return "Hault"
        if self.next_state:
            return self.next_state
        return None

    async def exit_actions(self) -> None:
        await super().exit_actions()
        self.next_state = None


def run():
    """Run test."""


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
    run()
