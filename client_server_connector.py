#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Server Client

"Client Server Client. Handles server connected to. Has network state and can handle events."

# Programmed by CoolCat467

__title__ = 'Client Server Client'
__author__ = 'CoolCat467'
__version__ = '0.0.0'

from connection import TCPAsyncSocketConnection
from gears import Timer
from events import EventHandler, Event, log_active_exception
from client_networking import ClientNetwork

class ClientServerConnector(TCPAsyncSocketConnection, Timer, EventHandler):# pylint: disable=R0901
    "Game client server connector"
    __slots__ = ('net', 'addr', 'runner_state')
    gear_type = 'client'
    min_delay = 0
    timeout = 20
    # typecheck: Another file has errors: Desktop/Python/Projects/Azul/client_networking.py
    def __init__(self, server, name: str, addr, runner_state) -> None:
        "Needs server/runner, gear name, and read write heads for socket."
        TCPAsyncSocketConnection.__init__(self)
        Timer.__init__(self, server, name, -1)

        self.addr = addr
        self.net = ClientNetwork(self)
        self.runner_state = runner_state

    server = property(lambda self: self.bot, doc='Server')

    async def start(self) -> None:
        if await self.connect_server():
            await self.net.initialize_state()
        return await super().start()

    async def connect_server(self) -> bool:
        "Connect to server at self.addr. Return True on success."
        try:
            await self.connect(self.addr, timeout=self.timeout)
        except (OSError, IOError, TimeoutError) as ex:
            self.submit_event(
                Event(
                    'connect_server_state',
                    message_type='error',
                    text=ex
                )
            )
            log_active_exception()
            return False
        return True

    async def tick(self) -> bool:
        "Have self.net think and return if it's active state is None."
        await self.net.think()
        return self.net.active_state is None

    async def net_stop(self) -> None:
        "Await self.net.stop if it exists"
        if hasattr(self.net, 'stop'):
            await self.net.stop()

    def gear_shutdown(self) -> None:
        self.close()
        super().gear_shutdown()

    def on_stop(self) -> None:
        "Remove this gear"
        self.bot.remove_gear(self.name)

    def submit_event(self, event) -> int:
        "Submit an event to runner"
        return self.bot.submit_event(event)

    async def read_event(self, event) -> None:
        "Give event to network statemachine"
        if hasattr(self.net, 'read_event'):
            await self.net.read_event(event)

def run():
    "Run example"

if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.')
    run()
