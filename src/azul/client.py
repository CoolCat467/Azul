"""Azul Client."""

from __future__ import annotations

import contextlib

# Programmed by CoolCat467
# Hide the pygame prompt
import os
import sys
from os import path
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import trio
from pygame.locals import K_ESCAPE, KEYUP, QUIT, RESIZABLE, WINDOWRESIZED
from pygame.rect import Rect

from azul import conf, lang, objects, sprite
from azul.component import Component, ComponentManager, Event
from azul.statemachine import AsyncState, AsyncStateMachine
from azul.vector import Vector2

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "True"
if os.environ["PYGAME_HIDE_SUPPORT_PROMPT"]:
    import pygame
del os


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__title__ = "Azul Client"
__author__ = "CoolCat467"
__version__ = "2.0.0"

SCREEN_SIZE = Vector2(800, 600)
FPS = 30
# FPS = 60
VSYNC = True
# PORT = server.PORT

ROOT_FOLDER: Final = Path(__file__).absolute().parent
DATA_FOLDER: Final = ROOT_FOLDER / "data"
FONT_FOLDER: Final = ROOT_FOLDER / "fonts"

FONT = FONT_FOLDER / "RuneScape-UF-Regular.ttf"


class AzulClient(sprite.GroupProcessor, AsyncStateMachine):
    """Gear Runner and Layered Dirty Sprite group handler."""

    def __init__(self) -> None:
        """Initialize azul client."""
        sprite.GroupProcessor.__init__(self)
        AsyncStateMachine.__init__(self)

        self.add_states(
            (
                HaltState(),
                AzulInitialize(),
            ),
        )

    @property
    def running(self) -> bool:
        """Boolean of if state machine is running."""
        return self.active_state is not None

    async def raise_event(self, event: Event[Any]) -> None:
        """Raise component event in all groups."""
        if self.active_state is None:
            return
        manager = getattr(self.active_state, "manager", None)
        assert isinstance(manager, ComponentManager | None)
        if manager is None:
            return
        await manager.raise_event(event)


class AzulState(AsyncState[AzulClient]):
    """Azul Client Asynchronous base class."""

    __slots__ = ("id", "manager")

    def __init__(self, name: str) -> None:
        """Initialize azul state."""
        super().__init__(name)

        self.id: int = 0
        self.manager = ComponentManager(self.name)


class HaltState(AzulState):
    """Halt state to set state to None so running becomes False."""

    def __init__(self) -> None:
        """Initialize halt state."""
        super().__init__("Halt")

    async def check_conditions(self) -> None:
        """Set active state to None."""
        await self.machine.set_state(None)


class ClickDestinationComponent(Component):
    """Component that will use targeting to go to wherever you click on the screen."""

    __slots__ = ("selected",)
    outline = pygame.color.Color(255, 220, 0)

    def __init__(self) -> None:
        """Initialize click destination component."""
        super().__init__("click_dest")

        self.selected = False

    def bind_handlers(self) -> None:
        """Register PygameMouseButtonDown and tick handlers."""
        self.register_handlers(
            {
                "click": self.click,
                "drag": self.drag,
                "PygameMouseButtonDown": self.mouse_down,
                "tick": self.move_towards_dest,
                "init": self.cache_outline,
                "test": self.test,
            },
        )

    async def test(self, event: Event[object]) -> None:
        """Print out event data."""
        print(f"{event = }")

    async def cache_outline(self, _: Event[None]) -> None:
        """Precalculate outlined images."""
        image: sprite.ImageComponent = self.get_component("image")
        outline: sprite.OutlineComponent = image.get_component("outline")
        outline.precalculate_all_outlined(self.outline)

    async def update_selected(self) -> None:
        """Update selected."""
        image: sprite.ImageComponent = self.get_component("image")
        outline: sprite.OutlineComponent = image.get_component("outline")

        color = (None, self.outline)[int(self.selected)]
        outline.set_color(color)

        if not self.selected:
            movement: sprite.MovementComponent = self.get_component("movement")
            movement.speed = 0

    async def click(
        self,
        event: Event[sprite.PygameMouseButtonEventData],
    ) -> None:
        """Toggle selected."""
        if event.data["button"] == 1:
            self.selected = not self.selected

            await self.update_selected()

    async def drag(self, event: Event[None]) -> None:
        """Drag sprite."""
        if not self.selected:
            self.selected = True
            await self.update_selected()
        movement: sprite.MovementComponent = self.get_component("movement")
        movement.speed = 0

    async def mouse_down(
        self,
        event: Event[sprite.PygameMouseButtonEventData],
    ) -> None:
        """Target click pos if selected."""
        if not self.selected:
            return
        if event.data["button"] == 1:
            movement: sprite.MovementComponent = self.get_component("movement")
            movement.speed = 200
            target: sprite.TargetingComponent = self.get_component("targeting")
            target.destination = Vector2.from_iter(event.data["pos"])

    async def move_towards_dest(
        self,
        event: Event[sprite.TickEventData],
    ) -> None:
        """Move closer to destination."""
        target: sprite.TargetingComponent = self.get_component("targeting")
        await target.move_destination_time(event.data.time_passed)


class MrFloppy(sprite.Sprite):
    """Mr. Floppy test sprite."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize mr floppy sprite."""
        super().__init__("MrFloppy")

        self.add_components(
            (
                sprite.MovementComponent(),
                sprite.TargetingComponent(),
                ClickDestinationComponent(),
                sprite.ImageComponent(),
                sprite.DragClickEventComponent(),
            ),
        )

        movement = self.get_component("movement")
        targeting = self.get_component("targeting")
        image = self.get_component("image")

        movement.speed = 200

        # lintcheck: c-extension-no-member (I1101): Module 'pygame.surface' has no 'Surface' member, but source is unavailable. Consider adding this module to extension-pkg-allow-list if you want to perform analysis based on run-time introspection of living objects.
        floppy: pygame.surface.Surface = pygame.image.load(
            path.join("data", "mr_floppy.png"),
        )

        image.add_images(
            {
                0: floppy,
                # '1': pygame.transform.flip(floppy, False, True)
                1: pygame.transform.rotate(floppy, 270),
                2: pygame.transform.flip(floppy, True, True),
                3: pygame.transform.rotate(floppy, 90),
            },
        )

        anim = image.get_component("animation")
        anim.controller = self.controller((0, 1, 2, 3))

        image.set_image(0)
        self.visible = True

        self.location = SCREEN_SIZE / 2
        targeting.destination = self.location

        self.register_handler("drag", self.drag)

    @staticmethod
    def controller(
        image_identifiers: Sequence[str | int],
    ) -> Iterator[str | int | None]:
        """Animation controller."""
        cidx = 0
        while True:
            count = len(image_identifiers)
            if not count:
                yield None
                continue
            cidx = (cidx + 1) % count
            yield image_identifiers[cidx]

    async def drag(self, event: Event[sprite.DragEvent]) -> None:
        """Move by relative from drag."""
        if event.data.button != 1:
            return
        self.location += event.data.rel
        self.dirty = 1


class FPSCounter(objects.Text):
    """FPS counter."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize fps counter."""
        font = pygame.font.Font(FONT, 28)
        super().__init__("fps", font)

    async def on_tick(self, event: Event[sprite.TickEventData]) -> None:
        """Update text."""
        # self.text = f'FPS: {event.data["fps"]:.2f}'
        self.text = f"FPS: {event.data.fps:.0f}"

    async def update_loc(
        self,
        event: Event[dict[str, tuple[int, int]]],
    ) -> None:
        """Move to top left corner."""
        self.location = Vector2.from_iter(event.data["size"]) / 2 + (5, 5)

    def bind_handlers(self) -> None:
        """Register event handlers."""
        super().bind_handlers()
        self.register_handlers(
            {
                "tick": self.on_tick,
                "sprite_image_resized": self.update_loc,
            },
        )


class AzulInitialize(AzulState):
    """Initialize Azul."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize state."""
        super().__init__("initialize")

    def group_add(self, new_sprite: sprite.Sprite) -> None:
        """Add new sprite to group."""
        group = self.machine.get_group(self.id)
        assert group is not None, "Expected group from new group id"
        group.add(new_sprite)
        self.manager.add_component(new_sprite)

    async def entry_actions(self) -> None:
        """Create group and add mr floppy."""
        self.id = self.machine.new_group("test")
        floppy = MrFloppy()
        print(floppy)
        self.group_add(floppy)
        self.group_add(FPSCounter())

        await self.machine.raise_event(Event("init", None))

    async def exit_actions(self) -> None:
        """Remove group and unbind components."""
        self.machine.remove_group(self.id)
        self.manager.unbind_components()


def save_crash_img() -> None:
    """Save the last frame before the game crashed."""
    surface = pygame.display.get_surface().copy()
    # strTime = '-'.join(time.asctime().split(' '))
    # filename = f'Crash_at_{strTime}.png'
    filename = "screenshot.png"

    pygame.image.save(surface, path.join("screenshots", filename))
    del surface


async def async_run() -> None:
    """Run client."""
    global SCREEN_SIZE
    # global client
    config = conf.load_config(path.join("conf", "main.conf"))
    lang.load_lang(config["Language"]["lang_name"])

    screen = pygame.display.set_mode(
        tuple(SCREEN_SIZE),
        RESIZABLE,
        vsync=VSYNC,
    )
    pygame.display.set_caption(f"{__title__} v{__version__}")
    pygame.key.set_repeat(1000, 30)
    screen.fill((0xFF, 0xFF, 0xFF))

    client = AzulClient()

    background = pygame.image.load(
        path.join("data", "background.png"),
    ).convert()
    client.clear(screen, background)

    client.set_timing_threshold(1000 / FPS)

    await client.set_state("initialize")

    clock = pygame.time.Clock()

    while client.running:
        resized_window = False

        async with trio.open_nursery() as nursery:
            for event in pygame.event.get():
                # pylint: disable=undefined-variable
                if event.type == QUIT:
                    await client.set_state("Halt")
                elif event.type == KEYUP and event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))
                elif event.type == WINDOWRESIZED:
                    SCREEN_SIZE = Vector2(event.x, event.y)
                    resized_window = True
                sprite_event = sprite.convert_pygame_event(event)
                # print(sprite_event)
                nursery.start_soon(client.raise_event, sprite_event)
        await client.think()

        time_passed = clock.tick(FPS)

        await client.raise_event(
            Event(
                "tick",
                sprite.TickEventData(
                    time_passed / 1000,
                    clock.get_fps(),
                ),
            ),
        )

        if resized_window:
            screen.fill((0xFF, 0xFF, 0xFF))
            rects = [Rect((0, 0), tuple(SCREEN_SIZE))]
            client.repaint_rect(rects[0])
            rects.extend(client.draw(screen))
        else:
            rects = client.draw(screen)
        pygame.display.update(rects)
    client.clear_groups()


class Tracer(trio.abc.Instrument):
    """Tracer instrument."""

    __slots__ = ("_sleep_time",)

    def before_run(self) -> None:
        """Before run."""
        print("!!! run started")

    def _print_with_task(self, msg: str, task: trio.lowlevel.Task) -> None:
        """Print message with task name."""
        # repr(task) is perhaps more useful than task.name in general,
        # but in context of a tutorial the extra noise is unhelpful.
        print(f"{msg}: {task.name}")

    def task_spawned(self, task: trio.lowlevel.Task) -> None:
        """Task spawned."""
        self._print_with_task("### new task spawned", task)

    def task_scheduled(self, task: trio.lowlevel.Task) -> None:
        """Task scheduled."""
        self._print_with_task("### task scheduled", task)

    def before_task_step(self, task: trio.lowlevel.Task) -> None:
        """Before task step."""
        self._print_with_task(">>> about to run one step of task", task)

    def after_task_step(self, task: trio.lowlevel.Task) -> None:
        """After task step."""
        self._print_with_task("<<< task step finished", task)

    def task_exited(self, task: trio.lowlevel.Task) -> None:
        """Task exited."""
        self._print_with_task("### task exited", task)

    def before_io_wait(self, timeout: float) -> None:
        """Before IO wait."""
        if timeout:
            print(f"### waiting for I/O for up to {timeout} seconds")
        else:
            print("### doing a quick check for I/O")
        self._sleep_time = trio.current_time()

    def after_io_wait(self, timeout: float) -> None:
        """After IO wait."""
        duration = trio.current_time() - self._sleep_time
        print(f"### finished I/O check (took {duration} seconds)")

    def after_run(self) -> None:
        """After run."""
        print("!!! run finished")


def run() -> None:
    """Run asynchronous side of everything."""
    trio.run(async_run)  # , instruments=[Tracer()])


# save_crash_img()

if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.\n")

    # Make sure the game will display correctly on high DPI monitors on Windows.
    if sys.platform == "win32":
        # Exists on windows but not on linux or macos
        # Windows raises attr-defined
        # others say unused-ignore
        from ctypes import windll  # type: ignore[attr-defined,unused-ignore]

        with contextlib.suppress(AttributeError):
            windll.user32.SetProcessDPIAware()
        del windll

    try:
        pygame.init()
        run()
    finally:
        pygame.quit()
