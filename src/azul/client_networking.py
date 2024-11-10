#!/usr/bin/env python3
# Client Networking

"""Client Networking."""

# Programmed by CoolCat467

__title__ = "Client Networking"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import datetime
import json
import random
from typing import Final

from connection import Connection, TCPAsyncSocketConnection
from event_statetimer import EventAsyncState, StatorEventExtend
from events import Event, log_active_exception
from gears import AsyncStateMachine, StateTimerExitState

PROTO_VER: Final[int] = 0


class ClientNetwork(AsyncStateMachine, StatorEventExtend):
    """Client network."""

    __slots__ = ("client", "srv_host", "srv_port")

    def __init__(self, client):
        super().__init__()
        self.client = client

        self.add_state(StateTimerExitState())
        self.add_state(ConnectState())
        self.add_state(LoginState())

    async def initialize_state(self) -> None:
        """Set state to handshake."""
        await self.set_state("connect")

    async def think(self) -> None:
        """Think, but set state to hault on exception and log exception."""
        try:
            await super().think()
        except Exception:  # pylint: disable=broad-except
            log_active_exception()
            await self.set_state("Hault")

    async def stop(self) -> None:
        """Set state to hault."""
        await self.set_state("Hault")

    def submit_event(self, event):
        """Submit an event to runner."""
        return self.client.submit_event(event)


class NetworkState(EventAsyncState):
    """Network State."""

    __slots__ = ("client",)

    def __init__(self, name: str):
        super().__init__(name)
        self.machine: ClientNetwork

        self.client: TCPAsyncSocketConnection

    async def entry_actions(self) -> None:
        """Set self.client."""
        self.client = self.machine.client

    async def exit_actions(self) -> None:
        """Clear self.client."""
        del self.client

    def do_handshake(self, next_state: int) -> None:
        """Send handshake to server."""
        # typecheck: error: "TCPAsyncSocketConnection" has no attribute "addr"
        host, port = self.client.addr

        buffer = Connection()

        buffer.write_varint(0x00)
        buffer.write_varint(PROTO_VER)
        buffer.write_utf(host)
        buffer.write_ushort(port)

        buffer.write_varint(next_state)

        self.client.write_buffer(buffer)


class ConnectState(NetworkState):
    """Perform handshake and get status, then attempt login."""

    __slots__ = ("data",)

    def __init__(self):
        super().__init__("connect")

        self.data = {}

    async def entry_actions(self) -> None:
        await super().entry_actions()
        self.data = {}

    def send_error(self, text: str) -> None:
        """Send connect server state error."""
        self.machine.submit_event(
            Event("connect_server_state", message_type="error", text=text),
        )

    async def do_status(self, status_type: int) -> float | dict:
        """Perform status request and return response."""
        packet = Connection()
        packet.write_varint(status_type)

        token = random.randint(0, (1 << 63) - 1)
        if status_type == 0x01:
            packet.write_long(token)

        sent = datetime.datetime.now()
        self.client.write_buffer(packet)

        response = await self.client.read_buffer()
        received = datetime.datetime.now()

        if response.read_varint() != status_type:
            self.send_error("Received invalid ping packet from server.")
            raise OSError("Received invalid ping response packet from server.")

        if status_type == 0x00:
            return json.loads(response.read_utf())
        response_token = response.read_long()
        if response_token != token:
            msg = "Received mangled ping response packet"
            self.send_error(msg)
            raise OSError(
                msg + f" (expected token {token}, received {response_token})",
            )
        delta = received - sent
        return (
            delta.days * 24 * 60 * 60 + delta.seconds
        ) * 1000 + delta.microseconds / 1000

    async def do_actions(self):
        """Do status request with server."""
        self.do_handshake(0x01)
        json_data = await self.do_status(0x00)
        latency = await self.do_status(0x01)
        self.client.close()

        self.data = {"json": json_data, "latency": latency}

    async def check_conditions(self):
        def send_error(text="Server status json response is invalid"):
            self.send_error(text)
            return "Hault"

        if "version" not in self.data["json"]:
            return send_error()
        if "protocol" not in self.data["json"]["version"]:
            return send_error()
        if self.data["json"]["version"]["protocol"] > PROTO_VER:
            return send_error("Client is outdated!")

        print(f"{self.data=}")
        return "login"


class LoginState(NetworkState):
    """Login state."""

    __slots__: tuple = ()

    def __init__(self):
        super().__init__("login")

    async def entry_actions(self):
        await super().entry_actions()

        print("logging in")
        await self.client.connect_server()
        self.do_handshake(0x02)

    async def check_conditions(self):
        return "Hault"


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
