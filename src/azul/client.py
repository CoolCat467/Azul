"""Game Client."""

# Programmed by CoolCat467

# Copyright (C) 2023-2024  CoolCat467
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__title__ = "Game Client"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"
__version__ = "0.0.0"

import struct
import traceback
from typing import TYPE_CHECKING

import trio
from libcomponent import network
from libcomponent.base_io import StructFormat
from libcomponent.buffer import Buffer
from libcomponent.component import Event
from libcomponent.network_utils import ClientNetworkEventComponent

from azul.network_shared import (
    ADVERTISEMENT_IP,
    ADVERTISEMENT_PORT,
    ClientBoundEvents,
    ServerBoundEvents,
    decode_cursor_location,
    decode_int8_array,
    decode_numeric_uint8_counter,
    decode_tile_count,
    encode_cursor_location,
)
from azul.vector import Vector2

if TYPE_CHECKING:
    from mypy_extensions import u8

    from azul.state import Tile


async def read_advertisements(
    timeout: int = 3,  # noqa: ASYNC109
) -> list[tuple[str, tuple[str, int]]]:
    """Read server advertisements from network. Return tuples of (motd, (host, port))."""
    # Look up multicast group address in name server and find out IP version
    addrinfo = (await trio.socket.getaddrinfo(ADVERTISEMENT_IP, None))[0]

    with trio.socket.socket(
        family=trio.socket.AF_INET,  # IPv4
        type=trio.socket.SOCK_DGRAM,  # UDP
        proto=trio.socket.IPPROTO_UDP,
    ) as udp_socket:
        # SO_REUSEADDR: allows binding to port potentially already in use
        # Allow multiple copies of this program on one machine
        # (not strictly needed)
        udp_socket.setsockopt(
            trio.socket.SOL_SOCKET,
            trio.socket.SO_REUSEADDR,
            1,
        )

        await udp_socket.bind(("", ADVERTISEMENT_PORT))

        # # Tell the kernel that we are a multicast socket
        # udp_socket.setsockopt(trio.socket.IPPROTO_IP, trio.socket.IP_MULTICAST_TTL, 255)

        # socket.IPPROTO_IP works on Linux and Windows
        # # IP_MULTICAST_IF: force sending network traffic over specific network adapter
        # IP_ADD_MEMBERSHIP: join multicast group
        # udp_socket.setsockopt(
        # trio.socket.IPPROTO_IP, trio.socket.IP_MULTICAST_IF,
        # trio.socket.inet_aton(network_adapter)
        # )
        # udp_socket.setsockopt(
        # trio.socket.IPPROTO_IP,
        # trio.socket.IP_ADD_MEMBERSHIP,
        # struct.pack(
        # "4s4s",
        # trio.socket.inet_aton(group),
        # trio.socket.inet_aton(network_adapter),
        # ),
        # )
        group_bin = trio.socket.inet_pton(addrinfo[0], addrinfo[4][0])
        # Join group
        if addrinfo[0] == trio.socket.AF_INET:  # IPv4
            mreq = group_bin + struct.pack("=I", trio.socket.INADDR_ANY)
            udp_socket.setsockopt(
                trio.socket.IPPROTO_IP,
                trio.socket.IP_ADD_MEMBERSHIP,
                mreq,
            )
        else:  # IPv6
            mreq = group_bin + struct.pack("@I", 0)
            udp_socket.setsockopt(
                trio.socket.IPPROTO_IPV6,
                trio.socket.IPV6_JOIN_GROUP,
                mreq,
            )

        host = ""
        buffer = b""
        with trio.move_on_after(timeout):
            buffer, address = await udp_socket.recvfrom(512)
            host, _port = address
        # print(f"{buffer = }")
        # print(f"{address = }")

        response: list[tuple[str, tuple[str, int]]] = []

        start = 0
        for _ in range(1024):
            ad_start = buffer.find(b"[AD]", start)
            if ad_start == -1:
                break
            ad_end = buffer.find(b"[/AD]", ad_start)
            if ad_end == -1:
                break
            start_block = buffer.find(b"[AZUL]", ad_end)
            if start_block == -1:
                break
            start_end = buffer.find(b"[/AZUL]", start_block)
            if start_end == -1:
                break

            start = start_end

            motd = buffer[start_block + 10 : start_end].decode("utf-8")
            raw_port = buffer[ad_start + 4 : ad_end].decode("utf-8")
            try:
                port = int(raw_port)
            except ValueError:
                continue
            response.append((motd, (host, port)))
        return response


class GameClient(ClientNetworkEventComponent):
    """Game Client Network Event Component.

    This class handles connecting to the game server, transmitting events
    to the server, and reading and raising incoming events from the server.
    """

    __slots__ = ("connect_event_lock", "running")

    def __init__(self, name: str) -> None:
        """Initialize GameClient."""
        super().__init__(name)

        # Five seconds until timeout is generous, but it gives server end wiggle
        # room.
        self.timeout = 5

        sbe = ServerBoundEvents
        self.register_network_write_events(
            {
                "encryption_response->server": sbe.encryption_response,
                "factory_clicked->server[write]": sbe.factory_clicked,
                "cursor_location->server[write]": sbe.cursor_location,
                "pattern_row_clicked->server[write]": sbe.pattern_row_clicked,
                "table_clicked->server[write]": sbe.table_clicked,
            },
        )
        cbe = ClientBoundEvents
        self.register_read_network_events(
            {
                cbe.encryption_request: "server->encryption_request",
                cbe.callback_ping: "server->callback_ping",
                cbe.initial_config: "server->initial_config",
                cbe.playing_as: "server->playing_as",
                cbe.game_over: "server->game_over",
                cbe.board_data: "server->board_data",
                cbe.pattern_data: "server->pattern_data",
                cbe.factory_data: "server->factory_data",
                cbe.cursor_data: "server->cursor_data",
                cbe.table_data: "server->table_data",
                cbe.cursor_movement_mode: "server->cursor_movement_mode",
                cbe.current_turn_change: "server->current_turn_change",
                cbe.cursor_position: "server->cursor_position",
            },
        )

        self.connect_event_lock = trio.Lock()
        self.running = False

    def bind_handlers(self) -> None:
        """Register event handlers."""
        super().bind_handlers()
        self.register_handlers(
            {
                "server->encryption_request": self.read_encryption_request,
                "server->callback_ping": self.read_callback_ping,
                "server->initial_config": self.read_initial_config,
                "server->playing_as": self.read_playing_as,
                "server->game_over": self.read_game_over,
                "server->board_data": self.read_board_data,
                "server->pattern_data": self.read_pattern_data,
                "server->factory_data": self.read_factory_data,
                "server->cursor_data": self.read_cursor_data,
                "server->table_data": self.read_table_data,
                "server->cursor_movement_mode": self.read_cursor_movement_mode,
                "server->current_turn_change": self.read_current_turn_change,
                "server->cursor_position": self.read_cursor_position,
                "client_connect": self.handle_client_connect,
                "network_stop": self.handle_network_stop,
                "game_factory_clicked": self.write_game_factory_clicked,
                "game_cursor_location_transmit": self.write_game_cursor_location_transmit,
                "game_pattern_row_clicked": self.write_game_pattern_row_clicked,
                "game_table_clicked": self.write_game_table_clicked,
            },
        )

    async def print_callback_ping(self, event: Event[bytearray]) -> None:
        """Print received `callback_ping` event from server.

        This event is used as a sort of keepalive heartbeat, because
        it stops the connection from timing out.
        """
        print(f"print_callback_ping {event = }")

    async def raise_disconnect(self, message: str) -> None:
        """Raise client_disconnected event with given message."""
        print(f"{self.__class__.__name__}: {message}")
        if not self.manager_exists:
            print(
                f"{self.__class__.__name__}: Manager does not exist, not raising disconnect event.",
            )
            return
        await self.raise_event(Event("client_disconnected", message))
        await self.close()
        assert self.not_connected

    async def handle_read_event(self) -> None:
        """Raise events from server.

        Can raise following exceptions:
          RuntimeError - Unhandled packet id
          network.NetworkStreamNotConnectedError - Network stream is not connected
          OSError - Stopped responding
          trio.BrokenResourceError - Something is wrong and stream is broken

        Shouldn't happen with write lock but still:
          trio.BusyResourceError - Another task is already writing data

        Handled exceptions:
          trio.ClosedResourceError - Stream is closed or another task closes stream
          network.NetworkTimeoutError - Timeout
          network.NetworkEOFError - Server closed connection
        """
        # print(f"{self.__class__.__name__}[{self.name}]: handle_read_event")
        if not self.manager_exists:
            return
        if self.not_connected:
            await self.raise_disconnect("Not connected to server.")
            return
        try:
            # print("handle_read_event start")
            event = await self.read_event()
        except trio.ClosedResourceError:
            self.running = False
            await self.close()
            print(f"[{self.name}] Socket closed from another task.")
            return
        except network.NetworkTimeoutError as exc:
            if self.running:
                self.running = False
                print(f"[{self.name}] NetworkTimeoutError")
                await self.close()
                traceback.print_exception(exc)
                await self.raise_disconnect(
                    "Failed to read event from server.",
                )
            return
        except network.NetworkStreamNotConnectedError as exc:
            self.running = False
            print(f"[{self.name}] NetworkStreamNotConnectedError")
            traceback.print_exception(exc)
            await self.close()
            assert self.not_connected
            raise
        except network.NetworkEOFError:
            self.running = False
            print(f"[{self.name}] NetworkEOFError")
            await self.close()
            await self.raise_disconnect(
                "Server closed connection.",
            )
            return

        await self.raise_event(event)

    async def handle_client_connect(
        self,
        event: Event[tuple[str, int]],
    ) -> None:
        """Have client connect to address specified in event."""
        if self.connect_event_lock.locked():
            raise RuntimeError("2nd client connect fired!")
        async with self.connect_event_lock:
            # Mypy does not understand that self.not_connected becomes
            # false after connect call.
            if not TYPE_CHECKING and not self.not_connected:
                raise RuntimeError("Already connected!")
            try:
                await self.connect(*event.data)
            except OSError as ex:
                traceback.print_exception(ex)
            else:
                self.running = True
                while not self.not_connected and self.running:
                    await self.handle_read_event()
                self.running = False

                await self.close()
                if self.manager_exists:
                    await self.raise_event(
                        Event("client_connection_closed", None),
                    )
                else:
                    print(
                        "manager does not exist, cannot send client connection closed event.",
                    )
                return
            await self.raise_disconnect("Error connecting to server.")

    async def read_initial_config(self, event: Event[bytearray]) -> None:
        """Read initial_config event from server."""
        buffer = Buffer(event.data)

        varient_play: u8 = buffer.read_value(StructFormat.BOOL)
        player_count: u8 = buffer.read_value(StructFormat.UBYTE)
        factory_count: u8 = buffer.read_value(StructFormat.UBYTE)
        current_turn: u8 = buffer.read_value(StructFormat.UBYTE)

        await self.raise_event(
            Event(
                "game_initial_config",
                (
                    varient_play,
                    player_count,
                    factory_count,
                    current_turn,
                ),
            ),
        )

    async def read_playing_as(self, event: Event[bytearray]) -> None:
        """Read playing_as event from server."""
        buffer = Buffer(event.data)

        playing_as: u8 = buffer.read_value(StructFormat.UBYTE)

        await self.raise_event(
            Event("game_playing_as", playing_as),
        )

    async def read_game_over(self, event: Event[bytearray]) -> None:
        """Read game_over event from server."""
        buffer = Buffer(event.data)

        winner: u8 = buffer.read_value(StructFormat.UBYTE)

        await self.raise_event(Event("game_winner", winner))
        self.running = False

    async def read_board_data(self, event: Event[bytearray]) -> None:
        """Read board_data event from server, reraise as `game_board_data`."""
        buffer = Buffer(event.data)

        player_id: u8 = buffer.read_value(StructFormat.UBYTE)
        array = decode_int8_array(buffer, (5, 5))

        await self.raise_event(Event("game_board_data", (player_id, array)))

    async def read_pattern_data(self, event: Event[bytearray]) -> None:
        """Read pattern_data event from server, reraise as `game_pattern_data`."""
        buffer = Buffer(event.data)

        player_id: u8 = buffer.read_value(StructFormat.UBYTE)
        row_id: u8 = buffer.read_value(StructFormat.UBYTE)
        tile_data = decode_tile_count(buffer)

        await self.raise_event(
            Event("game_pattern_data", (player_id, row_id, tile_data)),
        )

    async def read_factory_data(self, event: Event[bytearray]) -> None:
        """Read factory_data event from server, reraise as `game_factory_data`."""
        buffer = Buffer(event.data)

        factory_id: u8 = buffer.read_value(StructFormat.UBYTE)
        tiles = decode_numeric_uint8_counter(buffer)

        await self.raise_event(Event("game_factory_data", (factory_id, tiles)))

    async def read_cursor_data(self, event: Event[bytearray]) -> None:
        """Read cursor_data event from server, reraise as `game_cursor_data`."""
        buffer = Buffer(event.data)

        tiles = decode_numeric_uint8_counter(buffer)

        await self.raise_event(Event("game_cursor_data", tiles))

    async def read_table_data(self, event: Event[bytearray]) -> None:
        """Read table_data event from server, reraise as `game_table_data`."""
        buffer = Buffer(event.data)

        tiles = decode_numeric_uint8_counter(buffer)

        await self.raise_event(Event("game_table_data", tiles))

    async def read_cursor_movement_mode(self, event: Event[bytearray]) -> None:
        """Read cursor_movement_mode event from server, reraise as `game_cursor_set_movement_mode`."""
        buffer = Buffer(event.data)

        client_mode = buffer.read_value(StructFormat.BOOL)

        await self.raise_event(
            Event("game_cursor_set_movement_mode", client_mode),
        )

    async def read_current_turn_change(self, event: Event[bytearray]) -> None:
        """Read current_turn_change event from server, reraise as `game_pattern_current_turn_change`."""
        buffer = Buffer(event.data)

        pattern_id: u8 = buffer.read_value(StructFormat.UBYTE)
        await self.raise_event(
            Event("game_pattern_current_turn_change", pattern_id),
        )

    async def read_cursor_position(self, event: Event[bytearray]) -> None:
        """Read current_turn_change event from server, reraise as `game_cursor_set_destination`."""
        location = decode_cursor_location(event.data)
        unit_location = Vector2.from_iter(x / 0xFFF for x in location)

        await self.raise_event(
            Event("game_cursor_set_destination", unit_location),
        )

    async def write_game_factory_clicked(
        self,
        event: Event[tuple[int, Tile]],
    ) -> None:
        """Write factory_clicked event to server."""
        factory_id, tile = event.data
        buffer = Buffer()

        buffer.write_value(StructFormat.UBYTE, factory_id)
        buffer.write_value(StructFormat.UBYTE, tile)

        await self.raise_event(Event("factory_clicked->server[write]", buffer))

    async def write_game_cursor_location_transmit(
        self,
        event: Event[Vector2],
    ) -> None:
        """Write cursor_location_transmit event to server."""
        scaled_location = event.data

        x, y = map(int, (scaled_location * 0xFFF).floored())
        buffer = encode_cursor_location((x, y))

        await self.raise_event(Event("cursor_location->server[write]", buffer))

    async def write_game_pattern_row_clicked(
        self,
        event: Event[tuple[int, Vector2]],
    ) -> None:
        """Write factory_clicked event to server."""
        row_id, location = event.data
        buffer = Buffer()

        buffer.write_value(StructFormat.UBYTE, row_id)
        buffer.write_value(StructFormat.UBYTE, int(location.x))
        buffer.write_value(StructFormat.UBYTE, int(location.y))

        await self.raise_event(
            Event("pattern_row_clicked->server[write]", buffer),
        )

    async def write_game_table_clicked(
        self,
        event: Event[Tile],
    ) -> None:
        """Write table_clicked event to server."""
        tile = event.data
        buffer = Buffer()

        buffer.write_value(StructFormat.UBYTE, tile)

        await self.raise_event(Event("table_clicked->server[write]", buffer))

    async def handle_network_stop(self, event: Event[None]) -> None:
        """Send EOF if connected and close socket."""
        if self.not_connected:
            return
        self.running = False
        try:
            await self.send_eof()
        finally:
            await self.close()
        assert self.not_connected

    def __del__(self) -> None:
        """Print debug message."""
        print(f"del {self.__class__.__name__}")
