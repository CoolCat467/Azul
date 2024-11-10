#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sprite - Sprite Module

"Sprite"

# Programmed by CoolCat467

__title__ = 'Sprite'
__author__ = 'CoolCat467'
__version__ = '0.0.0'


from typing import Any, cast, Generic

from pygame.rect import Rect
from pygame.color import Color
from pygame.surface import Surface
from pygame import mask
from pygame.sprite import DirtySprite, LayeredDirty, LayeredUpdates

from location import Location
from component import Event, Component, ComponentManager


class Sprite(DirtySprite, ComponentManager):
    "Both Dirty Sprite and Component Manager"
    __slots__ = ('__location', 'rect')
    def __init__(self, name: str) -> None:
        DirtySprite.__init__(self)
        ComponentManager.__init__(self, name)
        
        self.rect = Rect(0, 0, 0, 0)
        self.__location: Location = Location(self.rect)
    
    def __get_location(self) -> Location:
        return self.__location
    def __set_location(self, value: tuple[int, int]) -> None:
        self.__location.x = value[0]
        self.__location.y = value[1]
    
    location = property(__get_location, __set_location, doc='Location')

    def __get_image_dims(self) -> tuple[int, int]:
        "Return size of internal rectangle"
        return self.rect.size
    def __set_image_dims(self, value: tuple[int, int]) -> None:
        "Set internal rectangle size while keeping self.location intact."
        prev_size = self.rect.size
        pre_loc = self.location.conv_ints()
        self.rect.size = value
                
        if self.rect.center == pre_loc:
            return
        
        rel = tuple(self.rect.center - pre_loc)
        
        self.location = pre_loc
        
        if prev_size == (0, 0) or self.rect.size == (0, 0):
            return
        
    image_dims = property(__get_image_dims, __set_image_dims, doc='Image dimentions')
    
    def __get_image(self) -> Surface | None:
        "Return surface of this sprite"
        return self.__image
    def __set_image(self, image: Surface | None) -> None:
        "Set surface and update image_dims"
        self.__image = image
        if not image is None:
            self.image_dims = image.get_size()
        self.dirty = 1
    
    image = property(__get_image, __set_image, doc='Image property auto-updating dimentions.')
    
    ##### Extra
    def is_selected(self, position: tuple[int, int]) -> bool:
        "Return True if visible, collision with point, and topmost at point"
        
        if not self.visible:
            return False
        if not self.rect.collidepoint(position):
            return False
        
        for group in self.groups():
            assert isinstance(group, LayeredUpdates), "Group must have get_sprites_at"
            sprites_at = group.get_sprites_at(position)
            if not sprites_at:
                continue
            top = sprites_at[-1]
            if top != self:
                return False
        return True


# Monkey-patch LayeredDirty to properly support __class_getitem__
LayeredDirty.__class_getitem__ = lambda x: LayeredDirty  # type: ignore[attr-defined]


class Group(LayeredDirty[Sprite], ComponentManager):
    "Group of Layered Dirty Sprites"
    __slots__ = ()
    def __init__(self,
                 name: str,
                 *sprites: Sprite,
                 **kwargs: Any) -> None:
        LayeredDirty.__init__(self, *sprites, **kwargs)  # type: ignore[type-var]
        ComponentManager.__init__(self, name)
    
    def group_add(self, sprite: Sprite, layer: int | None=None) -> None:
        "Only add sprite to render group, not to component."
        super().add_internal(sprite, layer)  # type: ignore[arg-type]
    
    def add_internal(self, sprite: Sprite, layer: int | None=None) -> None:
        super().add_internal(sprite, layer)  # type: ignore[arg-type]
        if isinstance(sprite, Component):
            super().add_component(sprite)
    
    def remove_internal(self, sprite: Sprite) -> None:
        super().remove_internal(sprite)
        if isinstance(sprite, Component):
            super().remove_component(sprite.name)


class Click(Component):
    "Raise `click` and `click_end` events on sprite when clicked"
    __slots__ = ('selected',)
    def __init__(self) -> None:
        super().__init__('click')
        
        self.selected = False
        
        self.add_handler('MouseButtonDown', self.handle_mouse_down)
        self.add_handler('MouseButtonUp', self.handle_mouse_up)
    
    async def handle_mouse_down(self, event: Event[tuple[int, int]]) -> None:
        "Handle mouse down events"
        if self.manager is None:
            return
        if self.manager.is_selected(event['pos']):
            self.selected = True
            await self.manager(Event('click', event))
        elif self.selected:
            self.selected = False
            await self.manager(Event('click_stop', event))
    
    async def handle_mouse_up(self, event: Event[Any]) -> None:
        "Handle mouse up events"
        if self.selected and self.manager is not None:
            self.selected = False
            # types: operator error: "ComponentManager" not callable
            await self.manager(Event('click_stop', event))
# types:          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class Draggable(Component):
    "Make Sprite Draggable"
    __slots__ = ('active',)
    def __init__(self) -> None:
        super().__init__('dragable')
        
        self.active = False
        
        # types: attr-defined error: "Draggable" has no attribute "add_handler"
        self.add_handler('drag', self.drag_start)
        # types: ^^^^^^^
        # types: attr-defined error: "Draggable" has no attribute "add_handler"
        self.add_handler('drag_stop', self.drag_end)
        # types: ^^^^^^^
        # types: attr-defined error: "Draggable" has no attribute "add_handler"
        self.add_handler('MouseMotion', self.handle_mouse_motion)
    # types: ^^^^^^^^^^^
    
    async def drag_start(self, event: Event[Any]) -> str:
        "Start dragging"
        self.active = True
        return 'break'
    
    async def drag_end(self, event: Event[Any]) -> str:
        "Start dragging"
        self.active = False
        return 'break'
    
    async def handle_mouse_motion(self, event: Event[tuple[int, int]]) -> None:
        "Handle mouse motion events"
        if not self.active or self.manager is None:
            return
        assert isinstance(self.manager, Sprite)
        # types: index error: Value of type "Event[tuple[int, int]]" is not indexable
        self.manager.location += event['rel']
        # types:                 ^^^^^^^^^^^^
        self.manager.dirty = 1


class PressHoldDrag(Component):
    "Raise drag events when held down"
    __slots__ = ()
    def __init__(self) -> None:
        super().__init__('press_hold_drag')
        
        # types: attr-defined error: "PressHoldDrag" has no attribute "add_handler"
        self.add_handler('click', self.start_click)
        # types: ^^^^^^^
        # types: attr-defined error: "PressHoldDrag" has no attribute "add_handler"
        self.add_handler('click_stop', self.stop_click)
    # types: ^^^^^^^^^^^
    
    async def start_click(self, event: Event[Any]) -> None:
        if self.manager is None:
            # types: unreachable error: Statement is unreachable
            return
        # types: ^
        # types: operator error: "ComponentManager" not callable
        # types: call-arg error: Missing positional argument "data" in call to "Event"
        await self.manager(Event('drag'))
    # types:  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    async def stop_click(self, event: Event[Any]) -> None:
        if self.manager is None:
            # types: unreachable error: Statement is unreachable
            return
        # types: ^
        # types: operator error: "ComponentManager" not callable
        # types: call-arg error: Missing positional argument "data" in call to "Event"
        await self.manager(Event('drag_stop'))
# types:      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class ToggleDrag(Component):
    "Raise drag events when held down"
    __slots__ = ('active',)
    def __init__(self) -> None:
        super().__init__('toggle_drag')
        
        self.active = False
        
        # types: attr-defined error: "ToggleDrag" has no attribute "add_handler"
        self.add_handler('click', self.handle_click)
        # types: ^^^^^^^
        # types: attr-defined error: "ToggleDrag" has no attribute "add_handler"
        self.add_handler('WindowLeave', self.handle_win_leave)
    # types: ^^^^^^^^^^^
    
    async def update(self) -> None:
        "Raise drag or drag_stop events depending on state"
        if self.manager is None:
            # types: unreachable error: Statement is unreachable
            return
        # types: ^
        if self.active:
            # types: operator error: "ComponentManager" not callable
            # types: call-arg error: Missing positional argument "data" in call to "Event"
            await self.manager(Event('drag'))
        # types:  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        else:
            # types: operator error: "ComponentManager" not callable
            # types: call-arg error: Missing positional argument "data" in call to "Event"
            await self.manager(Event('drag_stop'))
    # types:      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    async def handle_click(self, event: Event[Any]) -> None:
        "Toggle active on click event"
        self.active = not self.active
        await self.update()
    
    async def handle_win_leave(self, event: Event[Any]) -> None:
        "If active, stop dragging"
        if self.active:
            self.active = False
            await self.update()


class Outline(Component):
    "Outline sprite"
    __slots__ = ('active', 'mask_threshold')
    def __init__(self) -> None:
        super().__init__('outline')
        
        self.active = False
        self.mask_threshold = 0x7f
        
        # types: attr-defined error: "Outline" has no attribute "add_handler"
        self.add_handler('outline', self.outline_handler)
    # types: ^^^^^^^^^^^
    
    @staticmethod
    def _get_outline(surface: Surface,
                     size: int,
                     color: Color,
                     mask_threshold: int) -> Surface:
        "Outline surface"
        w, h = surface.get_size()
        
        diameter = size * 2
        surf = Surface((w+diameter, h+diameter)).convert_alpha()
        surf.fill(Color(0, 0, 0, 0))
        
        surf.lock()
        for ox, oy in mask.from_surface(surface, mask_threshold).outline():
            for x in range(diameter+1):
                for y in range(diameter+1):
                    surf.set_at((ox+x, oy+y), color)
        surf.unlock()
        surf.blit(surface, (size, size))
        return surf
    
    @staticmethod
    def _revert_outline(surface: Surface, color: Color) -> Surface:
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
        final.blit(surf, (0,0), area=area)
        return surf
    
    async def outline_handler(self, event: Event[bool | int | Color]) -> str:
        # types: index error: Value of type "Event[bool | int | Color]" is not indexable
        if event['enable'] != self.active and isinstance(self.manager, Sprite):
            # types: ^^^^^
            if not self.active:
                new = self._get_outline(
                    self.manager.image,
                    # types: index error: Value of type "Event[bool | int | Color]" is not indexable
                    event['size'],
                    # types: ^^^^
                    # types: index error: Value of type "Event[bool | int | Color]" is not indexable
                    event['color'],
                    # types: ^^^^^
                    self.mask_threshold
                )
                self.manager.image = new
            else:
                new = self._revert_outline(
                    self.manager.image,
                    # types: index error: Value of type "Event[bool | int | Color]" is not indexable
                    event['color']
                # types: ^^^^^^^^^
                )
                self.manager.image = new
            # types: index error: Value of type "Event[bool | int | Color]" is not indexable
            self.active = event['enable']
        # types:          ^^^^^^^^^^^^^^^
        return 'break'


class DragOutline(Component):
    "Enable outline while dragging"
    __slots__ = ('color', 'size')
    def __init__(self) -> None:
        super().__init__('drag_outline')
        
        self.color = Color(255, 0, 0)
        self.size = 2
        
        # types: attr-defined error: "DragOutline" has no attribute "add_handler"
        self.add_handler('drag', self.start_drag)
        # types: ^^^^^^^
        # types: attr-defined error: "DragOutline" has no attribute "add_handler"
        self.add_handler('drag_stop', self.stop_drag)
    # types: ^^^^^^^^^^^
    
    async def start_drag(self, event: Event[Any]) -> None:
        if self.manager is None:
            # types: unreachable error: Statement is unreachable
            return
        # types: ^
        # types: operator error: "ComponentManager" not callable
        await self.manager(Event('outline', {
            'enable': True,
            'color': self.color,
            'size': self.size
        }))
    async def stop_drag(self, event: Event[Any]) -> None:
        if self.manager is None:
            # types: unreachable error: Statement is unreachable
            return
        # types: ^
        # types: operator error: "ComponentManager" not callable
        # types: note: Another file has errors: /home/samuel/Desktop/Python/Projects/Azul/Original Azul (copy)/location.py
        await self.manager(Event('outline', {
            'enable': False,
            'color': self.color,
        }))

def run() -> None:
    "Run test of module"





if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.\n')
    run()
