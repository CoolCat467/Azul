#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Azul Client

"Azul Client"

# Programmed by CoolCat467

from typing import Iterator, Optional
from os import path

# Hide the pygame prompt
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'
del os

import trio

import pygame
from pygame.locals import *

from vector import Vector2
from statemachine import AsyncState, AsyncStateMachine
from component import Component, Event, ComponentManager
import sprite
import objects
import conf
import lang

__title__ = 'Azul Client'
__author__ = 'CoolCat467'
__version__ = '2.0.0'

SCREEN_SIZE = Vector2(800, 600)
FPS = 30
##FPS = 60
VSYNC = True
##PORT = server.PORT

class AzulState(AsyncState):
    "Azul Client Asynchronous base class"
    machine: 'AzulClient'
    __slots__ = ('id', 'manager')
    def __init__(self, name: str) -> None:
        super().__init__(name)

        self.id: int = 0
        self.manager = ComponentManager(self.name)

class HaltState(AzulState):
    "Halt state to set state to None so running becomes False"
    def __init__(self) -> None:
        super().__init__('Halt')

    async def check_conditions(self) -> None:
        "Set active state to None."
        await self.machine.set_state(None)

class ClickDestinationComponent(Component):
    "Component that will use targeting to go to wherever you click on the screen"
    __slots__ = ('selected',)
    outline = pygame.color.Color(255, 220, 0)
    def __init__(self) -> None:
        super().__init__('click_dest')

        self.selected = False

    def bind_handlers(self) -> None:
        "Register PygameMouseButtonDown and tick handlers"
        self.register_handlers({
            'click': self.click,
            'drag': self.drag,
            'PygameMouseButtonDown': self.mouse_down,
            'tick': self.move_towards_dest,
            'init': self.cache_outline,
            'test': self.test,
        })

    async def test(self, event: Event) -> None:
        print(f'{event = }')

    async def cache_outline(self, _: Event) -> None:
        "Precalculate outlined images"
        image: sprite.ImageComponent = self.get_component('image')
        outline: sprite.OutlineComponent = image.get_component('outline')
        outline.precalculate_all_outlined(self.outline)

    async def update_selected(self) -> None:
        "Update selected"
        image: sprite.ImageComponent = self.get_component('image')
        outline: sprite.OutlineComponent = image.get_component('outline')

        color = (None, self.outline)[int(self.selected)]
        outline.set_color(color)

        if not self.selected:
            movement: sprite.MovementComponent = self.get_component('movement')
            movement.speed = 0

    async def click(self, event: Event) -> None:
        "Toggle selected"
        if event.data['button'] == 1:
            self.selected = not self.selected

            await self.update_selected()

    async def drag(self, event: Event) -> None:
        "Drag sprite"
        if not self.selected:
            self.selected = True
            await self.update_selected()
        movement: sprite.MovementComponent = self.get_component('movement')
        movement.speed = 0

    async def mouse_down(self, event: Event) -> None:
        "Target click pos if selected"
        if not self.selected:
            return
        if event.data['button'] == 1:
            movement: sprite.MovementComponent = self.get_component('movement')
            movement.speed = 200
            target: sprite.TargetingComponent = self.get_component('targeting')
            target.destination = Vector2.from_iter(event.data['pos'])

    async def move_towards_dest(self, event: Event) -> None:
        "Move closer to destination"
        target: sprite.TargetingComponent = self.get_component('targeting')
        target.move_destination_time(event.data['time_passed'])

class MrFloppy(sprite.Sprite):
    "Mr. Floppy test sprite"
    __slots__ = ()
    def __init__(self) -> None:
        super().__init__('MrFloppy')

        self.add_components((
            sprite.MovementComponent(),
            sprite.TargetingComponent(),
            ClickDestinationComponent(),
            sprite.ImageComponent(),
            sprite.DragClickEventComponent(),
        ))

        movement = self.get_component('movement')
        targeting = self.get_component('targeting')
        image = self.get_component('image')

        movement.speed = 200

        # lintcheck: c-extension-no-member (I1101): Module 'pygame.surface' has no 'Surface' member, but source is unavailable. Consider adding this module to extension-pkg-allow-list if you want to perform analysis based on run-time introspection of living objects.
        floppy: pygame.surface.Surface = pygame.image.load(path.join('data', 'mr_floppy.png'))

        image.add_images({
            0: floppy,
##            '1': pygame.transform.flip(floppy, False, True)
            1: pygame.transform.rotate(floppy, 270),
            2: pygame.transform.flip(floppy, True, True),
            3: pygame.transform.rotate(floppy, 90),
        })

        anim = image.get_component('animation')
        anim.controller = self.controller((0, 1, 2, 3))

        image.set_image(0)
        self.visible = True

        self.location = SCREEN_SIZE/2
        targeting.destination = self.location

        self.register_handler('drag', self.drag)

    def controller(self, image_identifiers: list[str | int]) -> Iterator[Optional[str]]:
        "Animation controller"
        cidx = 0
        while True:
            count = len(image_identifiers)
            if not count:
                yield None
                continue
            cidx = (cidx + 1) % count
            yield image_identifiers[cidx]

    async def drag(self, event: Event) -> None:
        "Move by relative from drag"
        if event.data['button'] != 1:
            return
        sprite_component: sprite.Sprite = self.get_component('sprite')
        sprite_component.location += event.data['rel']
        sprite_component.dirty = 1

class FPSCounter(objects.Text):
    "FPS counter"
    __slots__ = ()
    def __init__(self) -> None:
        font = pygame.font.Font('data/RuneScape-UF-Regular.ttf', 28)
        super().__init__('fps', font)

    async def on_tick(self, event: Event) -> None:
        "Update text"
##        self.text = f'FPS: {event.data["fps"]:.2f}'
        self.text = f'FPS: {event.data["fps"]:.0f}'

    async def update_loc(self, event: Event) -> None:
        "Move to top left corner"
        self.location = Vector2.from_iter(event.data['size'])/2+(5, 5)

    def bind_handlers(self) -> None:
        super().bind_handlers()
        self.register_handlers({
            'tick': self.on_tick,
            'sprite_image_resized': self.update_loc,
        })

class AzulInitialize(AzulState):
    "Initialize Azul"
    __slots__ = ()
    def __init__(self) -> None:
        super().__init__('initialize')

    def group_add(self, new_sprite: sprite.Sprite) -> None:
        group = self.machine.get_group(self.id)
        assert group is not None, "Expected group from new group id"
        group.add(new_sprite)
        self.manager.add_component(new_sprite)

    async def entry_actions(self) -> None:
        self.id = self.machine.new_group('test')
        floppy = MrFloppy()
        print(floppy)
        self.group_add(floppy)
        self.group_add(FPSCounter())

        await self.machine.raise_event(Event(
            'init'
        ))

    async def exit_actions(self) -> None:
        self.machine.remove_group(self.id)
        self.manager.unbind_components()

class AzulClient(sprite.GroupProcessor, AsyncStateMachine):
    "Gear Runner and Layered Dirty Sprite group handler"
    def __init__(self) -> None:
        sprite.GroupProcessor.__init__(self)
        AsyncStateMachine.__init__(self)

        self.add_states((
            HaltState(),
            AzulInitialize(),
        ))

    @property
    def running(self) -> bool:
        "Boolean of if state machine is running."
        return self.active_state is not None

    async def raise_event(self, event: Event) -> None:
        "Raise component event in all groups"
        if self.active_state is None:
            return
        if self.active_state.manager is None:
            return
        await self.active_state.manager.raise_event(event)

def save_crash_img() -> None:
    "Save the last frame before the game crashed."
    surface = pygame.display.get_surface().copy()
##    strTime = '-'.join(time.asctime().split(' '))
##    filename = f'Crash_at_{strTime}.png'
    filename = 'screenshot.png'

    pygame.image.save(surface, path.join('screenshots', filename))
    del surface

async def async_run() -> None:
    "Run client"
    global SCREEN_SIZE
##    global client
    CONFIG = conf.load_config(path.join('conf', 'main.conf'))
    LANGUAGE = lang.load_lang(CONFIG['Language']['lang_name'])

    screen = pygame.display.set_mode(tuple(SCREEN_SIZE), RESIZABLE, vsync=VSYNC)
    pygame.display.set_caption(f'{__title__} v{__version__}')
    pygame.key.set_repeat(1000, 30)
    screen.fill((0xff, 0xff, 0xff))

    client = AzulClient()

    background = pygame.image.load(path.join('data', 'background.png')).convert()
    client.clear(screen, background)

    client.set_timing_treshold(1000/FPS)

    await client.set_state('initialize')

    clock = pygame.time.Clock()

    while client.running:
        resized_window = False

        async with trio.open_nursery() as nursery:
            for event in pygame.event.get():
                # pylint: disable=undefined-variable
                if event.type == QUIT:
                    await client.set_state('Halt')
                elif event.type == KEYUP and event.key == K_ESCAPE:
                    pygame.event.post(
                        pygame.event.Event(QUIT)
                    )
                elif event.type == WINDOWRESIZED:
                    SCREEN_SIZE = Vector2(event.x, event.y)
                    resized_window = True
                sprite_event = sprite.convert_pygame_event(event)
##                print(sprite_event)
                nursery.start_soon(client.raise_event, sprite_event)
        await client.think()

        time_passed = clock.tick(FPS)

        await client.raise_event(Event(
            'tick',
            {'time_passed': time_passed/1000,
             'fps': clock.get_fps()}
        ))

        if resized_window:
            screen.fill((0xff, 0xff, 0xff))
            rects = [Rect((0, 0), SCREEN_SIZE)]
            client.repaint_rect(rects[0])
            rects.extend(client.draw(screen))
        else:
            rects = client.draw(screen)
        pygame.display.update(rects)
    client.clear_groups()

class Tracer(trio.abc.Instrument):
    def before_run(self):
        print("!!! run started")

    def _print_with_task(self, msg, task):
        # repr(task) is perhaps more useful than task.name in general,
        # but in context of a tutorial the extra noise is unhelpful.
        print(f"{msg}: {task.name}")

    def task_spawned(self, task):
        self._print_with_task("### new task spawned", task)

    def task_scheduled(self, task):
        self._print_with_task("### task scheduled", task)

    def before_task_step(self, task):
        self._print_with_task(">>> about to run one step of task", task)

    def after_task_step(self, task):
        self._print_with_task("<<< task step finished", task)

    def task_exited(self, task):
        self._print_with_task("### task exited", task)

    def before_io_wait(self, timeout):
        if timeout:
            print(f"### waiting for I/O for up to {timeout} seconds")
        else:
            print("### doing a quick check for I/O")
        self._sleep_time = trio.current_time()

    def after_io_wait(self, timeout):
        duration = trio.current_time() - self._sleep_time
        print(f"### finished I/O check (took {duration} seconds)")

    def after_run(self):
        print("!!! run finished")


def run() -> None:
    "Run asynchronous side of everything"

    trio.run(async_run)#, instruments=[Tracer()])
##    save_crash_img()

if __name__ == '__main__':
    print(f'{__title__} v{__version__}\nProgrammed by {__author__}.\n')

    # Make sure the game will display correctly on high DPI monitors on Windows.
    import platform
    if platform.system() == 'Windows':
        from ctypes import windll# type: ignore
        try:
            windll.user32.SetProcessDPIAware()
        except AttributeError:
            pass
        del windll
    del platform

    try:
        pygame.init()
        run()
    finally:
        pygame.quit()
