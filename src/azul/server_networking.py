#!/usr/bin/env python3
# Server Networking

"""Server Networking"""

# Programmed by CoolCat467

__title__ = "Server Networking"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import json
import math
from typing import Final

from connection import Connection as Buffer
from event_statetimer import EventAsyncState, StatorEventExtend
from events import log_active_exception
from gears import AsyncStateMachine, StateTimerExitState

PROTOCOL: Final[int] = 0

##class Test(EventAsyncState):
##    "Test state"
##    __slots__ = ('client',)
##    def __init__(self):
##        super().__init__('test')
##        self.client = None
##
##    async def entry_actions(self) -> None:
##        "Set self.client"
##        # typecheck: Another file has errors: Desktop/Python/Projects/Azul/connection.py
##        # typecheck: error: "AsyncStateMachine" has no attribute "client"
##        self.client = self.machine.client
##
##    async def check_conditions(self) -> Union[str, None]:
##        "Read data from server"
##        text = (await self.client.reader.readuntil(b';')).decode('utf-8')
##        print(self.client.name+': '+text)
##        if 'stop' in text.lower():
##            self.client.stop_server()
##            self.client.submit_event(Event('Hault'))
##            return 'Hault'
##        if 'gears' in text.lower():
##            self.client.write_ascii(repr(self.client.bot.gears))
##        self.client.submit_event(
##            Event('chat_message', client=self.client.name,
##                  message=text, msg_enc='utf-8')
##            )
##        return None
##
##    async def on_event(self, event) -> None:
##        "Relay chat messages"
##        if event.type == 'chat_message':
##            if event['client'] == self.client.name:
##                return
##            msg = event['client'] + ': ' + event['message'][:-1]
##            self.client.write_ascii(msg)
##            return


class NetworkState(EventAsyncState):
    """Network State"""

    __slots__ = ("client",)

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.client = None

    # pylint: disable=unused-private-member
    def __get_proto_ver(self) -> int:
        """Return machine's protocol version"""
        # typecheck: Another file has errors: Desktop/Python/Projects/Azul/connection.py
        # typecheck: error: "AsyncStateMachine" has no attribute "proto_ver"
        return self.machine.proto_ver

    def __set_proto_ver(self, version: int) -> None:
        """Set machine's protocol version."""
        # typecheck: error: "AsyncStateMachine" has no attribute "proto_ver"
        self.machine.proto_ver = version

    proto_ver = property(
        __get_proto_ver,
        __set_proto_ver,
        doc="Machine's protocol version",
    )

    async def entry_actions(self) -> None:
        """Set self.client"""
        # typecheck: error: "AsyncStateMachine" has no attribute "client"
        self.client = self.machine.client

    async def exit_actions(self) -> None:
        """Clear self.client"""
        self.client = None


class Handshake(NetworkState):
    """Handshake state"""

    __slots__ = ("request_next",)
    state_id = 0x00

    def __init__(self):
        super().__init__("handshake")

        self.request_next = self.state_id

    async def do_actions(self) -> None:
        """Handle handshake buffer"""
        try:
            buffer = await self.client.read_buffer()
        except OSError:
            self.request_next = math.inf
            return

        state = buffer.read_varint()
        if state == self.state_id:
            self.proto_ver = buffer.read_varint()
            # typecheck: error: "AsyncStateMachine" has no attribute "srv_host"
            self.machine.srv_host = buffer.read_utf()
            # typecheck: error: "AsyncStateMachine" has no attribute "srv_port"
            self.machine.srv_port = buffer.read_ushort()
            self.request_next = buffer.read_varint()

    async def check_conditions(self) -> str | None:
        """Return next state"""
        if self.request_next in {0x00, 0x01, 0x02}:
            return (None, "status", "login")[self.request_next]
        return "Hault"


class Status(NetworkState):
    """Status state"""

    __slots__ = ("types_used", "disconnect")
    state_id = 0x01

    def __init__(self):
        super().__init__("status")

        self.types_used = {0x00: 0, 0x01: 0}
        self.disconnect = False

    async def do_actions(self) -> None:
        """Handle status requests"""
        try:
            buffer = await self.client.read_buffer()
        except OSError:
            self.disconnect = True
            return

        status_type = buffer.read_varint()
        if status_type not in {0x00, 0x01}:
            self.disconnect = True
            return
        self.types_used[status_type] += 1

        if status_type == 0x00 and sum(self.types_used.values()) > 1:
            self.disconnect = True
            return

        packet = Buffer()
        packet.write_varint(status_type)
        if status_type == 0x00:
            print(f"[{self.client.name}][Status Request]")
            packet.write_utf(json.dumps(self.client.server.get_status()))
        else:
            ping_token = buffer.read_long()
            print(f"[{self.client.name}][Ping Request]")
            packet.write_long(ping_token)
            self.disconnect = True

        self.client.write_buffer(packet)

    # typecheck: error: Missing return statement
    async def check_conditions(self) -> str | None:
        """Hault if disconnect"""
        if self.disconnect:
            return "Hault"
        return None


class Login(NetworkState):
    """Login state"""

    def __init__(self):
        super().__init__("login")

    async def check_conditions(self) -> str | None:
        """Hault if disconnect"""
        return "Hault"


class ServerClientNetwork(AsyncStateMachine, StatorEventExtend):
    """Server client network"""

    __slots__ = ("client", "proto_ver", "srv_host", "srv_port")

    def __init__(self, client):
        super().__init__()
        self.client = client

        self.proto_ver = -math.inf
        self.srv_host = ""
        self.srv_port = 0

        ##        self.add_state(Test())
        self.add_state(StateTimerExitState())
        self.add_state(Handshake())
        self.add_state(Status())
        self.add_state(Login())

    async def initialize_state(self) -> None:
        """Set state to handshake"""
        await self.set_state("handshake")

    async def think(self) -> None:
        """Think, but set state to hault on exception and log exception"""
        try:
            await super().think()
        except Exception:  # pylint: disable=broad-except
            log_active_exception()
            await self.set_state("Hault")

    def submit_event(self, event):
        """Submit an event to runner"""
        return self.client.submit_event(event)


def run():
    """Run"""


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
    run()
