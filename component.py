#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Component - Components instead of chaotic class hierarchy mess

"Component system module"

# Programmed by CoolCat467

__title__ = 'Component'
__author__ = 'CoolCat467'
__version__ = '0.0.0'

from typing import Any, Callable, TypeVar, cast, Iterable, Awaitable, Optional

import functools

import trio

class Event:
    "Event"
    __slots__ = ('name', 'data', 'level')
    def __init__(self,
                 name: str,
                 data: Optional[dict] = None,
                 levels: int = 0) -> None:
        self.name = name
        self.data = data if data is not None else {}
        self.level = levels

    def __repr__(self) -> str:
        "Return representation of self"
        items = {x: getattr(self, x) for x in self.__slots__ if not x.startswith('_')}
        return f'<{self.__class__.__name__} {items}>'

    def pop_level(self) -> bool:
        "Travel up one level and return if should continue or not"
        self.level = max(0, self.level - 1)
        return self.level > 0

def _get_real_attr(attr: str) -> str:
    "Remove hidden tags from attribute"
    while attr[0] == '_':
        attr = attr[1:]
    return attr

class Component:
    "Component base class"
    __slots__ = ('name', '__manager')
    def __init__(self, name: str) -> None:
        self.name = name
        self.__manager: Optional['ComponentManager'] = None

    def __repr__(self) -> str:
        "Return representation of self"
        return f'{self.__class__.__name__}({self.name!r})'

    @property
    def manager(self) -> 'ComponentManager':
        "ComponentManager if bound to one, otherwise raise AttributeError"
        if self.__manager is None:
            raise AttributeError(f'No component manager bound for {self.name}')
        return self.__manager

    def _unbind(self) -> None:
        "If you use this you are evil. This is only for ComponentManagers!"
        self.__manager = None

    @property
    def manager_exists(self) -> bool:
        "Return if manager is bound or not"
        return self.__manager is not None

    def register_handler(self,
                         event_name: str,
                         handler_coro: Callable[[Event], Awaitable[None]]) -> None:
        "Register handler with bound component manager"
        self.manager.register_handler(event_name, handler_coro)#, self.name)

    def register_handlers(self,
                          handlers: dict[str, Callable[[Event], Awaitable[None]]]) -> None:
        "Register multiple handler Coroutines"
        for name, coro in handlers.items():
            self.register_handler(name, coro)

    def bind_handlers(self) -> None:
        "Add handlers in subclass."

    def bind(self, manager: 'ComponentManager') -> None:
        "Bind self to manager"
        if self.manager_exists:
            raise RuntimeError(f'{self.name} component is already bound to {self.manager}')
        self.__manager = manager
        self.bind_handlers()

    async def raise_event(self, event: Event) -> None:
        "Raise event for bound manager"
        await self.manager.raise_event(event)

##    def raise_event_sync(self, event: Event) -> None:
##        "Raise event later for bound manager"
##        self.manager.raise_event_sync(event)

    def component_exists(self, component_name: str) -> bool:
        "Return if component exists in manager"
        return self.manager.component_exists(component_name)

    def components_exist(self, component_names: Iterable[str]) -> bool:
        "Return if all component names given exist in manager"
        return self.manager.components_exist(component_names)

    def get_component(self, component_name: str) -> 'Component':
        "Get Component from manager"
        return self.manager.get_component(component_name)

    def get_components(self, component_names: Iterable[str]) -> list['Component']:
        "Get Components from manager"
        return self.manager.get_components(component_names)

class ComponentManager(Component):
    "Component manager class. If own_name is set, adds self as component to self with name given"
    __slots__ = ('__event_handlers', '__components')
    def __init__(self, name: str, own_name: str = None) -> None:
        super().__init__(name)
        self.__event_handlers: dict[str, list[Callable[[Event], Awaitable[None]]]] = {}
        self.__components: dict[str, Component] = {}

        if own_name is not None:
            self.__add_self_as_component(own_name)

    def __repr__(self) -> str:
        return f'<ComponentManager Components: {tuple(self.__components.values())}>'

    def __add_self_as_component(self, name: str) -> None:
        "Add this manager as component to self without binding."
        if self.component_exists(name):
            raise ValueError(f'Component named "{name}" already exists!')
        self.__components[name] = self
        self.bind_handlers()

    def register_handler(self, event_name: str, handler_coro:
                         Callable[[Event], Awaitable[None]],
                         ) -> None:
        "Register handler_func as handler for event_name"
        if not event_name in self.__event_handlers:
            self.__event_handlers[event_name] = []
        self.__event_handlers[event_name].append(handler_coro)

    async def raise_event(self, event: Event) -> None:
        "Raise event for all components that have handlers registered"
        # Forward leveled events up; They'll come back to us soon enough.
        if self.manager_exists and event.pop_level():
            await super().raise_event(event)
            return

        # Call all registered handlers for this event
        if event.name in self.__event_handlers:
            async with trio.open_nursery() as nursery:
                for handler in self.__event_handlers[event.name]:
                    nursery.start_soon(handler, event)

        # Forward events to contained managers
        async with trio.open_nursery() as nursery:
            for component in self.get_all_components():
                # Skip self component if exists
                if component is self:
                    continue
                if isinstance(component, ComponentManager):
                    nursery.start_soon(component.raise_event, event)

    def add_component(self, component: Component) -> None:
        "Add component to this manager"
        assert isinstance(component, Component), "Must be component"
        if self.component_exists(component.name):
            raise ValueError(f'Component named "{component.name}" already exists!')
        component.bind(self)
        self.__components[component.name] = component

    def add_components(self, components: Iterable[Component]) -> None:
        "Add multiple components to this manager"
        for component in components:
            self.add_component(component)

    def component_exists(self, component_name: str) -> bool:
        "Return if component exists in this manager"
        return component_name in self.__components

    def components_exist(self, component_names: Iterable[str]) -> bool:
        "Return if all component names given exist in this manager"
        return all(self.component_exists(name) for name in component_names)

    def get_component(self, component_name: str) -> Component:
        "Return Component or raise ValueError"
        if not self.component_exists(component_name):
            raise ValueError(f'"{component_name}" component does not exist')
        return self.__components[component_name]

    def get_components(self, component_names: Iterable[str]) -> list[Component]:
        "Return iterable of components asked for or raise ValueError"
        return [self.get_component(name) for name in component_names]

    def list_components(self) -> list[str]:
        "Return list of components bound to this manager"
        return list(self.__components)

    def get_all_components(self) -> list[Component]:
        "Return all bound components"
        return list(self.__components.values())

    def unbind_components(self) -> None:
        "Unbind all components, allows things to get garbage collected."
        self.__event_handlers.clear()
        for component in iter(self.__components.values()):
            component._unbind()
        self.__components.clear()

    def __del__(self) -> None:
        self.unbind_components()

F = TypeVar('F', bound=Callable[..., Any])

def comps_must_exist(component_names: tuple[str, ...]) -> Callable[[F], F]:
    "Decorator for Components & ComponentManagers to ensure given components exist"
    def must_exist_decorator(func: F) -> F:
        "Wrap function and ensure component names exist."
        @functools.wraps(func)
        def must_exist_wrapper(self: Any, *args, **kwargs) -> Any:#type: ignore
            if not isinstance(self, (Component, ComponentManager)):
                raise TypeError(
                    'comps_must_exist must wrap a '\
                    'Component or ComponentManager function, '\
                    f'not "{type(self)}"!'
                )
            if not self.components_exist(component_names):
                raise RuntimeError(f'Not all components from {component_names} exist!')
            return func(self, *args, **kwargs)
        return cast(F, must_exist_wrapper)
    return must_exist_decorator

async def run_async() -> None:
    "Run test asynchronously"
    cat = ComponentManager('cat')
    sound_effect = Component('sound_effect')
    cat.add_component(sound_effect)
    print(cat)

def run() -> None:
    "Run test"
    trio.run(run_async)

if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.')
    run()
