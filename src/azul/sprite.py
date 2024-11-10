#!/usr/bin/env python3
# Client Sprite and Renderer

"""Client Sprite and Renderer"""

# Programmed by CoolCat467

from collections import deque
from collections.abc import Iterable, Iterator
from typing import ClassVar, Union

import trio
from component import Component, ComponentManager, Event
from pygame.color import Color
from pygame.event import Event as PygameEvent, event_name
from pygame.mask import Mask, from_surface as mask_from_surface
from pygame.rect import Rect
from pygame.sprite import DirtySprite, LayeredDirty, LayeredUpdates
from pygame.surface import Surface
from vector import Vector, Vector2

__title__ = "Client Sprite"
__author__ = "CoolCat467"
__version__ = "0.0.0"


class Location(Vector2):
    """Location bound to the center of a given rectangle"""

    __slots__ = ("_rect",)

    def __new__(
        cls,
        *args: int | float | complex,
        dtype: type = None,
    ) -> Union["Location", Vector2]:  # type: ignore
        """Super hack to return Vector2 if data type is not None, otherwise new Location."""
        if dtype is not None:
            new_vec = super().__new__(Vector2)
            new_vec.__init__(*args, dtype=dtype)  # type: ignore
            return new_vec
        return super().__new__(cls)

    def __init__(self, rect: Rect) -> None:
        """Initialize Location with rectangle."""
        self._rect = rect
        super().__init__(*self._rect.center, dtype=list)

    def __setitem__(
        self,
        index: int,
        value: int | float | complex,
        normal: bool = False,
    ) -> None:
        """Set item, but if not normal, updates rectangle as well."""
        super().__setitem__(index, value)
        if normal:
            return
        new_x, new_y = tuple(self)
        self._rect.center = new_x, new_y

    def __getitem__(self, index: int) -> int | float | complex:
        """Get item, but sets internal data at index to data from rectangle center first."""
        self.__setitem__(index, self._rect.center[index], normal=True)  # type: ignore
        return super().__getitem__(index)

    def normalize(self) -> Vector:
        """Raise NotImplemented, original is in place."""
        raise NotImplementedError

    def set_length(self, new_length: int | float | complex) -> Vector:
        """Raise NotImplemented, original is in place."""
        raise NotImplementedError


class Sprite(ComponentManager, DirtySprite):
    """Client sprite component"""

    __slots__ = ("rect", "__location", "__image", "sync_events", "mask")

    def __init__(self, name: str) -> None:
        ComponentManager.__init__(self, name, "sprite")
        DirtySprite.__init__(self)

        self.__image: Surface | None = None
        self.visible = False
        self.mask = None
        self.rect: Rect = Rect(0, 0, 0, 0)
        self.__location = Location(self.rect)

        self.sync_events: deque[Event] = deque()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Sprite {self.list_components()}>"

    # pylint: disable=unused-private-member
    def __get_location(self) -> Location:
        return self.__location

    def __set_location(self, value: list | tuple) -> None:
        # pylint: disable=invalid-name
        self.__location.x = value[0]
        self.__location.y = value[1]

    location = property(__get_location, __set_location, doc="Location")

    def __get_image_dims(self) -> tuple[int, int]:
        """Return size of internal rectangle"""
        return self.rect.size

    def __set_image_dims(self, value: tuple[int, int]) -> None:
        """Set internal rectangle size while keeping self.location intact."""
        prev_size = self.rect.size
        pre_loc = self.location.conv_ints()
        self.rect.size = value

        if self.rect.center == pre_loc:
            return

        rel = tuple(self.rect.center - pre_loc)

        self.location = pre_loc

        if prev_size == (0, 0) or self.rect.size == (0, 0):
            return
        self.sync_events.append(
            Event(
                "sprite_image_resized",
                {
                    "prev_size": prev_size,
                    "size": self.rect.size,
                    "rel": rel,
                },
            ),
        )

    image_dims = property(
        __get_image_dims,
        __set_image_dims,
        doc="Image dimensions",
    )

    def __get_image(self) -> Surface | None:
        """Return surface of this sprite"""
        return self.__image

    def __set_image(self, image: Surface | None) -> None:
        """Set surface and update image_dims"""
        self.__image = image
        if image is not None:
            self.image_dims = image.get_size()
        self.dirty = 1

    image = property(
        __get_image,
        __set_image,
        doc="Image property auto-updating dimensions.",
    )  # type: ignore

    ##### Extra
    def is_selected(self, position: tuple[int, int]) -> bool:
        """Return True if visible, collision with point, and topmost at point"""
        if not self.visible:
            return False
        if not self.rect.collidepoint(position):
            return False

        for group in self.groups():
            assert isinstance(
                group,
                LayeredUpdates,
            ), "Group must have get_sprites_at"
            sprites_at = group.get_sprites_at(position)
            if not sprites_at:
                continue
            top = sprites_at[-1]
            if top != self:
                return False
        return True

    async def raise_queued_sync_events(self, _: Event) -> None:
        """Raise queued sync events"""
        async with trio.open_nursery() as nursery:
            for event in (
                self.sync_events.popleft()
                for _ in range(len(self.sync_events))
            ):
                nursery.start_soon(self.raise_event, event)

    def bind_handlers(self) -> None:
        """Have tick raise queued sync events"""
        self.register_handler("tick", self.raise_queued_sync_events)


class ImageComponent(ComponentManager):
    """Allow sprite to use multiple images easily"""

    __slots__ = ("__surfaces", "__masks", "set_surface", "mask_threshold")

    def __init__(self) -> None:
        super().__init__("image")

        self.__surfaces: dict[int | str, Surface] = {}
        self.__masks: dict[int | str, Mask] = {}
        self.set_surface: int | str | None = None

        self.mask_threshold = 0x7F

        self.add_components(
            (
                AnimationComponent(),
                OutlineComponent(),
            ),
        )

    def _compute_mask(self, identifier: int | str) -> None:
        """Save mask for identifier"""
        self.__masks[identifier] = mask_from_surface(
            self.get_image(identifier),
            self.mask_threshold,
        )

    def add_image(self, identifier: int | str, surface: Surface) -> None:
        """Add image to internal database"""
        self.__surfaces[identifier] = surface
        self._compute_mask(identifier)

    def add_images(self, images: dict[int | str, Surface]) -> None:
        """Add images to internal database"""
        for ident, surf in images.items():
            self.add_image(ident, surf)

    def list_images(self) -> list[int | str]:
        """Return a list of saved image identifiers"""
        return list(self.__surfaces)

    def image_exists(self, identifier: int | str) -> bool:
        """Return if identifier exists in saved images"""
        return identifier in self.__surfaces

    def get_image(self, identifier: int | str) -> Surface:
        """Get saved image from identifier"""
        if not self.image_exists(identifier):
            raise ValueError(f'No image saved for identifier "{identifier}"')
        return self.__surfaces[identifier]

    def get_mask(self, identifier: int | str) -> Mask:
        """Get saved mask for saved image if exists, otherwise save and return"""
        if not self.image_exists(identifier):
            raise ValueError(f'No image saved for identifier "{identifier}"')
        if identifier not in self.__masks:
            self._compute_mask(identifier)
        return self.__masks[identifier]

    def set_image(self, identifier: int | str) -> None:
        """Set sprite component's image by identifier."""
        outline: OutlineComponent = self.get_component("outline")
        if outline.active and outline.mod not in str(identifier):
            identifier = outline.get_outline(identifier)

        if identifier == self.set_surface:
            return

        sprite: Sprite = self.manager.get_component("sprite")
        sprite.image = self.get_image(identifier)
        sprite.mask = self.get_mask(identifier)

        self.set_surface = identifier


class OutlineComponent(Component):
    """Add outline to sprite"""

    __slots__ = ("__active", "__color", "size")
    mod = "_outlined_"

    def __init__(self) -> None:
        super().__init__("outline")

        self.__active = False
        self.__color = Color(0xFF, 0xFF, 0xFF, 0)
        self.size = 5

    @property
    def active(self) -> bool:
        """Is active?"""
        return self.__active

    def set_color(self, color: Color | None) -> None:
        """Set color. If None, disable, otherwise enable."""
        prev = self.active
        self.__active = color is not None
        if not self.__active:
            if prev:
                search = str(self.manager.set_surface).split(self.mod)[0]
                for term in self.manager.list_images():
                    if term == search or str(term) == search:
                        break
                else:
                    return
                self.manager.set_image(term)
            return
        self.__color = color
        self.manager.set_image(self.manager.set_surface)

    def get_outline_discriptor(self, identifier: str | int) -> str:
        """Return outlined identifier for given original identifier"""
        return f"{identifier}{self.mod}{int(self.__color)}_{self.size}"

    def save_outline(self, identifier: str | int) -> None:
        """Save outlined version of given identifier image"""
        outlined = self.get_outline_discriptor(identifier)
        if self.manager.image_exists(outlined):
            return

        surface = self.manager.get_image(identifier)

        w, h = surface.get_size()

        diameter = self.size * 2
        surf = Surface((w + diameter, h + diameter)).convert_alpha()
        surf.fill(Color(0, 0, 0, 0))

        surf.lock()
        for ox, oy in self.manager.get_mask(identifier).outline():
            for x in range(diameter + 1):
                for y in range(diameter + 1):
                    surf.set_at((ox + x, oy + y), self.__color)
        surf.unlock()
        surf.blit(surface, (self.size, self.size))

        self.manager.add_image(outlined, surf)

    def get_outline(self, identifier: str | int) -> str:
        """Get saved outline effect identifier"""
        self.save_outline(identifier)
        return self.get_outline_discriptor(identifier)

    def precalculate_all_outlined(self, color: Color) -> None:
        """Precalculate all images outlined"""
        prev, self.__color = self.__color, color
        for image in self.manager.list_images():
            self.save_outline(image)
        self.__color = prev


class AnimationComponent(Component):
    """Allows sprite texture to be animated

    self.controller is an Iterator that yields
    either None or the name of a state in self.states

    update_every is float of how many seconds controller
    should be queried after. If zero, every tick.
    """

    __slots__ = ("controller", "update_every", "update_mod")

    def __init__(self) -> None:
        super().__init__("animation")

        def default() -> Iterator[None]:
            while True:
                yield self.manager.set_surface

        self.controller: Iterator[str | None] = default()

        self.update_every: float = 1.0
        self.update_mod: float = 0.0

    def fetch_controller_new_state(self) -> None:
        """Ask controller for new state"""
        return next(self.controller)

    async def tick(self, tick_event: Event) -> None:
        """Update controller if it's time to and update sprite image"""
        passed = tick_event.data["time_passed"]
        new, update = None, False
        if self.update_every == 0:
            new = self.fetch_controller_new_state()
            update = True
        else:
            updates, self.update_mod = divmod(
                self.update_mod + passed,
                self.update_every,
            )
            for _ in range(int(updates)):
                new = self.fetch_controller_new_state()
            update = updates > 0
        if update:
            self.manager.set_image(new)

    def bind_handlers(self) -> None:
        """Bind tick handler"""
        self.register_handler("tick", self.tick)


class MovementComponent(Component):
    """Component that moves sprite in direction of heading with speed."""

    __slots__ = ("heading", "speed")

    def __init__(self) -> None:
        super().__init__("movement")

        self.heading = Vector2(0, 0)
        self.speed = 0

    def point_toward(self, position: Iterable) -> None:
        """Change self.heading to point toward a given position."""
        sprite: Sprite = self.get_component("sprite")  # type: ignore
        self.heading = Vector2.from_points(
            sprite.location,
            position,
        ).normalized()

    def move_heading_distance(self, distance: float) -> None:
        """Move distance in heading direction."""
        sprite: Sprite = self.get_component("sprite")  # type: ignore
        change = self.heading * distance
        if change:
            sprite.location += change
            sprite.dirty = 1

    def move_heading_time(self, time_passed: float) -> None:
        """Move at speed in heading direction"""
        self.move_heading_distance(self.speed * time_passed)


class TargetingComponent(Component):
    """Sprite that moves toward a destination and then stops."""

    __slots__ = ("__destination",)

    def __init__(self) -> None:
        super().__init__("targeting")

        self.__destination = Vector2(0, 0)

    def update_heading(self) -> None:
        """Update the heading of the movement component."""
        movement: MovementComponent = self.get_component("movement")
        if self.to_destination == 0:
            movement.heading = Vector2(0, 0)
            return
        movement.heading = self.to_destination.normalized()

    def __set_destination(self, value: Iterable[int]) -> None:
        """Set destination as well as movement heading"""
        self.__destination = Vector2.from_iter(value)
        self.update_heading()

    def __get_destination(self) -> Vector2:
        """Get destination"""
        return self.__destination

    destination = property(
        __get_destination,
        __set_destination,
        doc="Target Destination",
    )

    @property
    def to_destination(self) -> Vector:
        """Return vector of self.location to self.destination"""
        sprite: Sprite = self.get_component("sprite")  # type: ignore
        return Vector2.from_points(sprite.location, self.destination)

    def move_destination_time(self, time_passed: float) -> None:
        """Move with time_passed"""
        sprite: Sprite
        movement: MovementComponent
        sprite, movement = self.get_components(("sprite", "movement"))  # type: ignore

        if sprite.location == self.destination:
            return

        travel_distance = min(
            self.to_destination.magnitude,
            movement.speed * time_passed,
        )

        if travel_distance > 0:
            movement.move_heading_distance(travel_distance)
            self.update_heading()  # Fix imprecision


class DragClickEventComponent(Component):
    """Raise drag events when motion, sprite visible, and sprite is top most."""

    __slots__ = ("pressed",)

    def __init__(self) -> None:
        super().__init__("drag_click_event")

        self.pressed: dict[int, bool] = {}

    async def press_start(self, event: Event) -> None:
        """Set pressed for event button if selected. Also raise Click events"""
        sprite: Sprite = self.get_component("sprite")

        if not sprite.is_selected(event.data["pos"]):
            await self.raise_event(Event("click_other", event.data))
            return

        self.pressed[event.data["button"]] = True

        await self.raise_event(Event("click", event.data))

    async def press_end(self, event: Event) -> None:
        """Unset pressed for event button"""
        self.pressed[event.data["button"]] = False

    async def motion(self, event: Event) -> None:
        """PygameMouseMotion event -> drag"""
        ##        sprite: Sprite = self.get_component('sprite')
        ##
        ##        if not sprite.is_selected(event.data['pos']):
        ##            self.pressed.clear()

        async with trio.open_nursery() as nursery:
            for button, pressed in self.pressed.items():
                if not pressed:
                    continue
                nursery.start_soon(
                    self.raise_event,
                    Event(
                        "drag",
                        {
                            "pos": event.data["pos"],
                            "rel": event.data["rel"],
                            "button": button,
                        },
                    ),
                )

    def bind_handlers(self) -> None:
        """Bind PygameMouseMotion"""
        ##        self.register_handler('PygameMouseMotion', self.motion)
        self.register_handlers(
            {
                "PygameMouseButtonDown": self.press_start,
                "PygameMouseButtonUp": self.press_end,
                "PygameMouseMotion": self.motion,
            },
        )


##class Group(ComponentManager, LayeredDirty):
##    def __init__(self) -> None:
##        ComponentManager.__init__(self, 'group')
##        LayeredDirty.__init__(self)


class GroupProcessor:
    """Gear Runner and Layered Dirty Sprite group handler"""

    ##    __slots__ = ('groups', 'group_names', 'new_gid', '_timing', '_clear')
    sub_renderer_class: ClassVar = LayeredDirty
    groups: dict[int, sub_renderer_class]

    def __init__(self) -> None:
        self.groups = {}
        self.group_names: dict[str, int] = {}
        self.new_gid = 0
        self._timing = 1000 / 80
        self._clear: tuple[Surface | None, Surface | None] = None, None

    def clear(self, screen: Surface, background: Surface) -> None:
        """Clear for all groups"""
        self._clear = screen, background
        for group in self.groups.values():
            group.clear(*self._clear)

    def set_timing_treshold(self, value: float) -> None:
        """set_timing_treshold for all groups"""
        self._timing = value
        for renderer in self.groups.values():
            renderer.set_timing_treshold(self._timing)

    def new_group(self, name: str = None) -> int:
        """Make a new group and return id"""
        if name is not None:
            self.group_names[name] = self.new_gid
        self.groups[self.new_gid] = self.sub_renderer_class()
        self.groups[self.new_gid].set_timing_treshold(self._timing)
        if self._clear[1] is not None:
            self.groups[self.new_gid].clear(*self._clear)
        self.new_gid += 1
        return self.new_gid - 1

    def remove_group(self, gid: int) -> None:
        """Remove group"""
        if gid in self.groups:
            for sprite in self.groups[gid].sprites():
                if isinstance(sprite, ComponentManager):
                    sprite.unbind_components()
            del self.groups[gid]
            for name, v_gid in self.group_names.items():
                if v_gid == gid:
                    # pylint: disable=unnecessary-dict-index-lookup
                    del self.group_names[name]
                    return

    def get_group(
        self,
        gid_name: str | int,
    ) -> sub_renderer_class | None:
        """Return group from group ID or name"""
        named = None
        if isinstance(gid_name, str):
            named = gid_name
            if gid_name in self.group_names:
                group_id = self.group_names[gid_name]
        else:
            group_id = gid_name
        if group_id in self.groups:
            return self.groups[group_id]
        if named is not None:
            del self.group_names[named]
        return None

    ##    def update(self, time_passed: float) -> None:
    ##        "Process gears and update sprites"
    ##        for group in self.groups.values():
    ##            group.update(time_passed)

    def draw(self, screen: Surface) -> list[Rect]:
        """Draw all groups"""
        rects = []
        for group in self.groups.values():
            rects.extend(group.draw(screen))
        return rects

    def repaint_rect(self, screen_rect: Rect) -> None:
        """Repaint rect for all groups"""
        for group in self.groups.values():
            group.repaint_rect(screen_rect)

    def clear_groups(self) -> None:
        """Clear all groups"""
        for group_id in tuple(self.groups):
            self.remove_group(group_id)

    def __del__(self) -> None:
        self.clear_groups()


def convert_pygame_event(event: PygameEvent) -> Event:
    """Convert Pygame Event to Component Event"""
    ##    data = event.dict
    ##    data['type_int'] = event.type
    ##    data['type'] = event_name(event.type)
    ##    return Event('pygame', data)
    return Event(f"Pygame{event_name(event.type)}", event.dict)


def run() -> None:
    """Run test"""


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
    run()
