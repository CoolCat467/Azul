#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Server Client

"Server Client. Handles clients connected. Has network state and can handle events."

# Programmed by CoolCat467

__title__ = 'Server Client'
__author__ = 'CoolCat467'
__version__ = '0.0.0'

import asyncio

from connection import TCPAsyncSocketConnection
from gears import Timer
from events import EventHandler
from server_networking import ServerClientNetwork

class GameServerClient(TCPAsyncSocketConnection, Timer, EventHandler):# pylint: disable=R0901
    "Game server client"
    __slots__ = ('net',)
    gear_type = 'client'
    min_delay = 0
    def __init__(self, server, name: str, reader, writer):
        "Needs server/runner, gear name, and read write heads for socket."
        TCPAsyncSocketConnection.__init__(self)
        self.reader = reader
        self.writer = writer
        Timer.__init__(self, server, name, -1)

        self.net = ServerClientNetwork(self)

    server = property(lambda self: self.bot, doc='Server')

    async def start(self) -> None:
        await self.net.initialize_state()
        return await super().start()

    async def tick(self) -> bool:
        "Have self.net think and return if it's active state is None."
        await self.net.think()
        return self.net.active_state is None

    def gear_shutdown(self) -> None:
        self.close()
        super().gear_shutdown()

    def on_stop(self) -> None:
        "Remove this gear"
        self.bot.remove_client(self.name)

    def stop_server(self) -> asyncio.Task:
        "Stop server by canceling coroutine"
        return self.submit_coro(self.bot.stop())

    def submit_event(self, event) -> int:
        "Submit an event to runner"
        return self.bot.submit_event(event)

    async def read_event(self, event) -> None:
        "Give event to network statemachine"
        if hasattr(self.net, 'on_event'):
            await self.net.on_event(event)

if __name__ == '__main__':
    print(f'{__title__}\nProgrammed by {__author__}.')
