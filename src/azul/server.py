#!/usr/bin/env python3
# Azul Server

"""Azul Server."""

# Programmed by CoolCat467

import asyncio
import multiprocessing
import socket
from concurrent.futures import ThreadPoolExecutor

from address_tools import find_ip, ip_type

##from server_room import Room, Liminal
from events import Event, EventLoop, EventLoopProcessor
from server_client import GameServerClient
from server_networking import PROTOCOL

__title__ = "Azul Server"
__author__ = "CoolCat467"
__version__ = "0.0.0"

LOCAL = True
PORT = 30654
MAX_ROOMS = 5
MAX_CLIENTS = MAX_ROOMS * 4


class GameServer(EventLoop):
    """Game Server Class."""

    __slots__ = ("port", "host", "server", "next_id", "ready", "num_clients")

    def __init__(self, port: int, host: str, loop):
        super().__init__(loop)

        self.port = port
        self.host = host
        self.server = None
        self.next_id = 0
        self.ready = False

        self.num_clients = 0

    def __repr__(self):
        return f"<{self.__class__.__name__} Game Server ({self.gears})>"

    def get_status(self) -> dict:  # pylint: disable=no-self-use
        """Return status for status pings."""
        ##        {
        ##    "version": {
        ##        "name": "1.8.7",
        ##        "protocol": 47
        ##    },
        ##    "players": {
        ##        "max": 100,
        ##        "online": 5,
        ##        "sample": [
        ##            {
        ##                "name": "thinkofdeath",
        ##                "id": "4566e69f-c907-48ee-8d71-d7ba5aa00d20"
        ##            }
        ##        ]
        ##    },
        ##    "description": {
        ##        "text": "Hello world"
        ##    },
        ##    "favicon": "data:image/png;base64,<data>"
        ##}
        return {
            "version": {
                "name": __version__,
                "protocol": PROTOCOL,
            },
            "description": {"text": __title__},
            "players": {"max": MAX_CLIENTS, "online": 0},
            ##            'rooms': {
            ##                'max': MAX_ROOMS,
            ##                'active': 0,
            ##                'joinable': [
            ##                    {
            ##                        'name': 'Test room',
            ##                        'id': 0,
            ##                        'open': False
            ##                    }
            ##                ]
            ##            }
        }

    async def bind(self):
        """Bind server."""
        host_ip_type = ip_type(self.host)
        if host_ip_type is None:
            host_ip_type = 4
        self.server = await asyncio.start_server(
            self.client_connect,
            host=self.host,
            port=self.port,
            family=socket.AF_INET if host_ip_type == 4 else socket.AF_INET6,
        )

    ##        self.loop = self.server.get_loop()

    def add_client(self, client_reader, client_writer):
        """Add client."""
        ##        writer_socket = client_writer.get_extra_info('socket')
        ##        writer_socket.settimeout(30)
        print(f"[Connect] client_{self.next_id}")
        client = GameServerClient(
            self,
            f"client_{self.next_id}",
            client_reader,
            client_writer,
        )

        self.add_gear(client)
        self.next_id += 1
        self.submit_event(Event("client_connect", name=client.name))
        self.num_clients += 1

    def remove_client(self, client_name):
        """Remove client."""
        print(f"[Disconnect] {client_name}")
        self.remove_gear(client_name)
        self.num_clients -= 1

    async def client_connect(self, client_reader, client_writer):
        """Client connects."""
        # reader = StreamReader
        # writer = StreamWriter
        self.add_client(client_reader, client_writer)

    async def start(self):
        """Start server."""
        await self.bind()
        proc = EventLoopProcessor(self)
        self.add_gear(proc)
        await self.server.start_serving()
        self.ready = True
        await self.run()

    async def run(self):
        """Run server."""
        async with self.server:
            try:
                await self.server.serve_forever()
            except asyncio.exceptions.CancelledError:
                pass
            finally:
                await self.stop()

    async def close(self) -> None:
        await super().close()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            for connection in self.server.sockets:
                connection.shutdown(socket.SHUT_RDWR)
                connection.close()

    async def stop(self):
        """Stop server."""
        self.server.close()
        await self.close()


def create_server(host: str, port: int, loop) -> GameServer:
    """Create server and return server and event loop."""
    return GameServer(port, host, loop)


def run(host, port):
    """Run server."""
    ##    global server

    loop = asyncio.new_event_loop()

    loop.set_default_executor(ThreadPoolExecutor(multiprocessing.cpu_count()))

    print(f"{host}:{port}")
    loop, server = create_server(host, port, loop)

    try:
        loop.run_until_complete(server.start())
    except KeyboardInterrupt:
        loop.run_until_complete(server.close())
    finally:
        # cancel all lingering tasks
        loop.close()
        print("Server closed")


if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.\n")
    run("localhost" if LOCAL else find_ip(), PORT)
