#!/usr/bin/env python3
# Keyboard

"""Keyboard Module"""

# Programmed by CoolCat467

__title__ = "Keyboard"
__author__ = "CoolCat467"
__version__ = "0.0.0"
__ver_major__ = 0
__ver_minor__ = 0
__ver_patch__ = 0


from component import ComponentManager


class Keyboard(ComponentManager):
    """Keyboard Componet"""

    def __init__(self):
        super().__init__("keyboard")


### OLD

import math


class Keyboard:
    """Keyboard object, handles keyboard input."""

    def __init__(self, key_down, key_up, **kwargs):
        ##        self.target = target
        ##        self.target.keyboard = self
        ##        self.target.is_pressed = self.is_pressed

        self.keys = {}  # Map of keyboard events to names
        self.actions = {}  # Map of keyboard event names to functions
        self.time = (
            {}
        )  # Map of names to time until function should be called again
        self.delay = (
            {}
        )  # Map of names to duration timer waits for function recalls
        self.active = {}  # Map of names to boolian of pressed or not

        self.key_events = {"down": key_down, "up": key_up}

        if kwargs:
            for name in kwargs:
                if not hasattr(kwargs[name], "__iter__"):
                    raise ValueError(
                        "Keyword arguments must be given as name=[key, self.target.function_name, delay]",
                    )
                if len(kwargs[name]) == 2:
                    key, function_name = kwargs[name]
                    delay = None
                elif len(kwargs[name]) == 3:
                    key, function_name, delay = kwargs[name]
                else:
                    raise ValueError
                self.add_listener(key, name)
                self.bind_action(name, function_name)

    def __repr__(self):
        ##        return 'Keyboard(%s)' % repr(self.target)
        return "<Keyboard>"

    def is_pressed(self, key):
        """Return True if <key> is pressed."""
        if key in self.active:
            return self.active[key]
        return False

    def add_listener(self, key: int, name: str):
        """Listen for key down events with event.key == key arguement and when that happens set self.actions[name] to true."""
        self.keys[key] = name  # key to name
        self.actions[name] = lambda: None  # name to function
        self.time[name] = 0  # name to time until function recall
        self.delay[name] = None  # name to function recall delay
        self.active[name] = False  # name to boolian of pressed

    ##    def get_function_from_target(self, function_name:str):
    ##        "Return function with name function_name from self.target"
    ##        if hasattr(self.target, function_name):
    ##            return getattr(self.target, function_name)
    ##        else:
    ##            return lambda: None

    def bind_action(self, name: str, function, delay=None):
        """Bind an event we are listening for to calling a function, can call multiple times if delay is not None."""
        ##        self.actions[name] = self.get_function_from_target(target_function_name)
        self.actions[name] = function
        self.delay[name] = delay

    def set_active(self, name: str, value: bool):
        """Set active value for key name <name> to <value>."""
        if name in self.active:
            self.active[name] = bool(value)
            if not value:
                self.time[name] = 0

    def set_key(self, key: int, value: bool, _nochar=False):
        """Set active value for key <key> to <value>"""
        if key in self.keys:
            self.set_active(self.keys[key], value)
        elif not _nochar:
            if key < 0x110000:
                self.set_key(chr(key), value, True)

    def read_event(self, event):
        """Handles an event."""
        if event.type == self.key_events["down"]:
            ##            print(f'key down: {event.key}')
            self.set_key(event.key, True)
        elif event.type == self.key_events["up"]:
            ##            print(f'key up: {event.key}')
            self.set_key(event.key, False)

    def read_events(self, events):
        """Handles a list of events."""
        for event in events:
            self.read_event(event)

    def process(self, time_passed):
        """Call functions based on pressed keys and time."""
        for name in self.active:
            if self.active[name]:
                self.time[name] = max(self.time[name] - time_passed, 0)
                if self.time[name] == 0:
                    self.actions[name]()
                    if self.delay[name] is not None:
                        self.time[name] = self.delay[name]
                    else:
                        self.time[name] = math.inf


def run():
    """Run"""


if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.")
    run()
