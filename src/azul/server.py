#!/usr/bin/env python3
# Checkers Game Server

"""Checkers Game Server."""

# Programmed by CoolCat467

from __future__ import annotations

# Copyright (C) 2023-2024  CoolCat467
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__title__ = "Server"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"
__version__ = "0.0.0"

import time
import traceback
from collections import deque
from functools import partial
from typing import TYPE_CHECKING, NoReturn

import trio
from libcomponent import network
from libcomponent.base_io import StructFormat
from libcomponent.buffer import Buffer
from libcomponent.component import (
    ComponentManager,
    Event,
    ExternalRaiseManager,
)
from libcomponent.network_utils import (
    ServerClientNetworkEventComponent,
    find_ip,
)

from azul.network_shared import (
    ADVERTISEMENT_IP,
    ADVERTISEMENT_PORT,
    DEFAULT_PORT,
    ClientBoundEvents,
    ServerBoundEvents,
)
from azul.state import State

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class ServerClient(ServerClientNetworkEventComponent):
    """Server Client Network Event Component.

    When clients connect to server, this class handles the incoming
    connections to the server in the way of reading and raising events
    that are transferred over the network.
    """

    __slots__ = ("client_id",)

    def __init__(self, client_id: int) -> None:
        """Initialize Server Client."""
        self.client_id = client_id
        super().__init__(f"client_{client_id}")

        self.timeout = 3

        cbe = ClientBoundEvents
        self.register_network_write_events(
            {
                "server[write]->encryption_request": cbe.encryption_request,
                "server[write]->callback_ping": cbe.callback_ping,
                "server[write]->initial_config": cbe.initial_config,
                "server[write]->game_over": cbe.game_over,
                "server[write]->playing_as": cbe.playing_as,
            },
        )
        sbe = ServerBoundEvents
        self.register_read_network_events(
            {
                sbe.encryption_response: f"client[{self.client_id}]->encryption_response",
            },
        )

    def bind_handlers(self) -> None:
        """Bind event handlers."""
        super().bind_handlers()
        self.register_handlers(
            {
                f"callback_ping->network[{self.client_id}]": self.handle_callback_ping,
                f"client[{self.client_id}]->encryption_response": self.handle_encryption_response,
                "initial_config->network": self.handle_initial_config,
                "game_over->network": self.handle_game_over,
                f"playing_as->network[{self.client_id}]": self.handle_playing_as,
            },
        )

    async def handle_game_over(self, event: Event[int]) -> None:
        """Read game over event and reraise as server[write]->game_over."""
        winner = event.data

        buffer = Buffer()

        buffer.write_value(StructFormat.UBYTE, winner)

        await self.write_event(Event("server[write]->game_over", buffer))

    async def handle_initial_config(
        self,
        event: Event[tuple[None, int]],
    ) -> None:
        """Read initial config event and reraise as server[write]->initial_config."""
        board_size, player_turn = event.data

        buffer = Buffer()

        ##        write_position(buffer, board_size)
        buffer.write_value(StructFormat.UBYTE, 0)
        buffer.write_value(StructFormat.UBYTE, player_turn)

        await self.write_event(Event("server[write]->initial_config", buffer))

    async def handle_playing_as(
        self,
        event: Event[int],
    ) -> None:
        """Read playing as event and reraise as server[write]->playing_as."""
        playing_as = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.UBYTE, playing_as)
        await self.write_event(Event("server[write]->playing_as", buffer))

    async def write_callback_ping(self) -> None:
        """Write callback_ping packet to client.

        Could raise the following exceptions:
          trio.BrokenResourceError: if something has gone wrong, and the stream
            is broken.
          trio.ClosedResourceError: if stream was previously closed

        Listed as possible but probably not because of write lock:
          trio.BusyResourceError: if another task is using :meth:`write`
        """
        buffer = Buffer()

        # Try to be as accurate with time as possible
        await self.wait_write_might_not_block()
        ns = int(time.time() * 1e9)
        # Use as many bits as time needs, write_buffer handles size for us.
        buffer.write(ns.to_bytes(-(-ns.bit_length() // 8), "big"))

        await self.write_event(Event("server[write]->callback_ping", buffer))

    async def handle_callback_ping(
        self,
        _: Event[None],
    ) -> None:
        """Reraise as server[write]->callback_ping."""
        await self.write_callback_ping()

    async def start_encryption_request(self) -> None:
        """Start encryption request and raise as `server[write]->encryption_request`."""
        await super().start_encryption_request()

        event = await self.read_event()
        if event.name != f"client[{self.client_id}]->encryption_response":
            raise RuntimeError(
                f"Expected encryption response, got but {event.name!r}",
            )
        await self.handle_encryption_response(event)


class GameServer(network.Server):
    """Checkers server.

    Handles accepting incoming connections from clients and handles
    main game logic via State subclass above.
    """

    __slots__ = (
        "actions_queue",
        "advertisement_scope",
        "client_count",
        "client_players",
        "internal_singleplayer_mode",
        "player_selections",
        "players_can_interact",
        "running",
        "state",
    )

    max_clients = 4

    def __init__(self, internal_singleplayer_mode: bool = False) -> None:
        """Initialize server."""
        super().__init__("GameServer")

        self.client_count: int
        self.state = State.new_game(0)

        self.client_players: dict[int, int] = {}
        self.players_can_interact: bool = False

        self.internal_singleplayer_mode = internal_singleplayer_mode
        self.advertisement_scope: trio.CancelScope | None = None
        self.running = False

    def bind_handlers(self) -> None:
        """Register start_server and stop_server."""
        self.register_handlers(
            {
                "server_start": self.start_server,
                "network_stop": self.stop_server,
                "server_send_game_start": self.handle_server_start_new_game,
            },
        )

    async def stop_server(self, event: Event[None] | None = None) -> None:
        """Stop serving and disconnect all NetworkEventComponents."""
        self.stop_serving()
        self.stop_advertising()

        close_methods: deque[Callable[[], Awaitable[object]]] = deque()
        for component in self.get_all_components():
            if isinstance(component, network.NetworkEventComponent):
                close_methods.append(component.close)
            print(f"stop_server {component.name = }")
            self.remove_component(component.name)
        async with trio.open_nursery() as nursery:
            while close_methods:
                nursery.start_soon(close_methods.popleft())
        self.running = False

    async def post_advertisement(
        self,
        udp_socket: trio.socket.SocketType,
        send_to_ip: str,
        hosting_port: int,
    ) -> None:
        """Post server advertisement packet."""
        motd = "Azul Game"
        advertisement = (
            f"[AD]{hosting_port}[/AD][AZUL]{motd}[/AZUL]"
        ).encode()
        # print("post_advertisement")
        await udp_socket.sendto(
            advertisement,
            (send_to_ip, ADVERTISEMENT_PORT),
        )

    def stop_advertising(self) -> None:
        """Cancel self.advertisement_scope."""
        if self.advertisement_scope is None:
            return
        self.advertisement_scope.cancel()

    async def post_advertisements(self, hosting_port: int) -> None:
        """Post lan UDP packets so server can be found."""
        self.stop_advertising()
        self.advertisement_scope = trio.CancelScope()

        # Look up multicast group address in name server and find out IP version
        addrinfo = (await trio.socket.getaddrinfo(ADVERTISEMENT_IP, None))[0]
        send_to_ip = addrinfo[4][0]

        with trio.socket.socket(
            family=trio.socket.AF_INET,  # IPv4
            type=trio.socket.SOCK_DGRAM,  # UDP
            proto=trio.socket.IPPROTO_UDP,  # UDP
        ) as udp_socket:
            # Set Time-to-live (optional)
            # ttl_bin = struct.pack('@i', MYTTL)
            # if addrinfo[0] == trio.socket.AF_INET: # IPv4
            # udp_socket.setsockopt(
            # trio.socket.IPPROTO_IP, trio.socket.IP_MULTICAST_TTL, ttl_bin)
            # else:
            # udp_socket.setsockopt(
            # trio.socket.IPPROTO_IPV6, trio.socket.IPV6_MULTICAST_HOPS, ttl_bin)
            with self.advertisement_scope:
                print("Starting advertisement posting.")
                while True:  # not self.can_start():
                    try:
                        await self.post_advertisement(
                            udp_socket,
                            send_to_ip,
                            hosting_port,
                        )
                    except OSError as exc:
                        traceback.print_exception(exc)
                        print(
                            f"{self.__class__.__name__}: Failed to post server advertisement",
                        )
                        break
                    await trio.sleep(1.5)
            print("Stopped advertisement posting.")

    @staticmethod
    def setup_teams_internal(client_ids: list[int]) -> dict[int, int]:
        """Return teams for internal server mode given sorted client ids."""
        players: dict[int, int] = {}
        for idx, client_id in enumerate(client_ids):
            if idx == 0:
                players[client_id] = 2
            else:
                players[client_id] = 0xFF  # Spectator
        return players

    @staticmethod
    def setup_teams(client_ids: list[int]) -> dict[int, int]:
        """Return teams given sorted client ids."""
        players: dict[int, int] = {}
        for idx, client_id in enumerate(client_ids):
            if idx < 2:
                players[client_id] = idx % 2
            else:
                players[client_id] = 0xFF  # Spectator
        return players

    def new_game_init(self) -> None:
        """Start new game."""
        self.client_players.clear()

        ##        pieces = generate_pieces(*self.board_size)
        self.state = State.new_game(self.client_count)

        # Why keep track of another object just to know client ID numbers
        # if we already have that with the components? No need!
        client_ids: set[int] = set()
        for component in self.get_all_components():
            if isinstance(component, ServerClient):
                client_ids.add(component.client_id)

        sorted_client_ids = sorted(client_ids)
        if self.internal_singleplayer_mode:
            self.client_players = self.setup_teams_internal(sorted_client_ids)
        else:
            self.client_players = self.setup_teams(sorted_client_ids)

        self.players_can_interact = True

    # "Implicit return in function which does not return"
    async def start_server(  # type: ignore[misc]
        self,
        event: Event[tuple[str | None, int]],
    ) -> NoReturn:
        """Serve clients."""
        print(f"{self.__class__.__name__}: Closing old server clients")
        await self.stop_server()
        print(f"{self.__class__.__name__}: Starting Server")
        self.client_count = 0

        host, port = event.data

        self.running = True
        async with trio.open_nursery() as nursery:
            # Do not post advertisements when using internal singleplayer mode
            if not self.internal_singleplayer_mode:
                nursery.start_soon(self.post_advertisements, port)
            # Serve runs forever until canceled
            nursery.start_soon(partial(self.serve, port, host, backlog=0))

    async def transmit_playing_as(self) -> None:
        """Transmit playing as."""
        async with trio.open_nursery() as nursery:
            for client_id, team in self.client_players.items():
                nursery.start_soon(
                    self.raise_event,
                    Event(f"playing_as->network[{client_id}]", team),
                )

    async def handle_server_start_new_game(self, event: Event[None]) -> None:
        """Handle game start."""
        ##        # Delete all pieces from last state (shouldn't be needed but still.)
        ##        async with trio.open_nursery() as nursery:
        ##            for piece_pos, _piece_type in self.state.get_pieces():
        ##                nursery.start_soon(
        ##                    self.raise_event,
        ##                    Event("delete_piece->network", piece_pos),
        ##                )

        # Choose which team plays first
        # Using non-cryptographically secure random because it doesn't matter
        self.new_game_init()

        ##        # Send create_piece events for all pieces
        ##        async with trio.open_nursery() as nursery:
        ##            for piece_pos, piece_type in self.state.get_pieces():
        ##                nursery.start_soon(
        ##                    self.raise_event,
        ##                    Event("create_piece->network", (piece_pos, piece_type)),
        ##                )

        await self.transmit_playing_as()

        # Raise initial config event with board size and initial turn.
        await self.raise_event(
            Event(
                "initial_config->network",
                (None, self.state.turn),
            ),
        )

    async def client_network_loop(self, client: ServerClient) -> None:
        """Network loop for given ServerClient.

        Could raise the following exceptions:
          trio.BrokenResourceError: if something has gone wrong, and the stream
            is broken.
          trio.ClosedResourceError: if stream was previously closed

        Probably couldn't raise because of write lock but still:
          trio.BusyResourceError: More than one task is trying to write
            to socket at once.
        """
        while not self.can_start() and not client.not_connected:
            try:
                await client.write_callback_ping()
            except (
                trio.BrokenResourceError,
                trio.ClosedResourceError,
                network.NetworkStreamNotConnectedError,
            ):
                print(f"{client.name} Disconnected in lobby.")
                return
        while not client.not_connected:
            event: Event[bytearray] | None = None
            try:
                await client.write_callback_ping()
                with trio.move_on_after(2):
                    event = await client.read_event()
            except network.NetworkTimeoutError:
                print(f"{client.name} Timeout")
                break
            except network.NetworkEOFError:
                print(f"{client.name} EOF")
                break
            except (
                trio.BrokenResourceError,
                trio.ClosedResourceError,
                RuntimeError,
            ):
                break
            except Exception as exc:
                traceback.print_exception(exc)
                break
            if event is not None:
                # print(f"{client.name} client_network_loop tick")
                # print(f"{client.name} {event = }")
                await client.raise_event(event)

    def can_start(self) -> bool:
        """Return if game can start."""
        if self.internal_singleplayer_mode:
            return self.client_count >= 1
        return self.client_count >= 2

    def game_active(self) -> bool:
        """Return if game is active."""
        return self.state.check_for_win() is None

    async def send_spectator_join_packets(
        self,
        client: ServerClient,
    ) -> None:
        """Send spectator start data."""
        print("send_spectator_join_packets")

        private_events_pocket = ComponentManager(
            f"private_events_pocket for {client.client_id}",
        )
        with self.temporary_component(private_events_pocket):
            with private_events_pocket.temporary_component(client):
                # Raise initial config event with board size and initial turn.
                await client.raise_event(
                    Event(
                        "initial_config->network",
                        (None, self.state.turn),
                    ),
                )

                await client.raise_event(
                    Event(f"playing_as->network[{client.client_id}]", 255),
                )

    async def handler(self, stream: trio.SocketStream) -> None:
        """Accept clients. Called by network.Server.serve."""
        if self.client_count == 0 and self.game_active():
            # Old game was running but everyone left, restart
            print("TODO: restart")
        new_client_id = self.client_count
        print(
            f"{self.__class__.__name__}: client connected [client_id {new_client_id}]",
        )
        self.client_count += 1

        can_start = self.can_start()
        game_active = self.game_active()
        # if can_start:
        # self.stop_serving()

        if self.client_count > self.max_clients:
            print(
                f"{self.__class__.__name__}: client disconnected, too many clients",
            )
            await stream.aclose()
            self.client_count -= 1
            return

        async with ServerClient.from_stream(
            new_client_id,
            stream=stream,
        ) as client:
            # Encrypt traffic
            await client.start_encryption_request()
            assert client.encryption_enabled

            if can_start and game_active:
                await self.send_spectator_join_packets(client)
            with self.temporary_component(client):
                if can_start and not game_active:
                    await self.raise_event(
                        Event("server_send_game_start", None),
                    )
                try:
                    await self.client_network_loop(client)
                finally:
                    print(
                        f"{self.__class__.__name__}: client disconnected [client_id {new_client_id}]",
                    )
                    self.client_count -= 1
        # ServerClient's `with` block handles closing stream.

    def __del__(self) -> None:
        """Debug print."""
        print(f"del {self.__class__.__name__}")
        super().__del__()


async def run_server(
    server_class: type[GameServer],
    host: str,
    port: int,
) -> None:
    """Run machine client and raise tick events."""
    async with trio.open_nursery() as main_nursery:
        event_manager = ExternalRaiseManager(
            "azul",
            main_nursery,
        )
        server = server_class()
        event_manager.add_component(server)

        await event_manager.raise_event(Event("server_start", (host, port)))
        while not server.running:
            print("Server starting...")
            await trio.sleep(1)

        print("\nServer running.")

        try:
            while server.running:  # noqa: ASYNC110  # sleep in while loop
                # Process background tasks in the main nursery
                await trio.sleep(0.01)
        except KeyboardInterrupt:
            print("\nClosing from keyboard interrupt.")
        await server.stop_server()
        server.unbind_components()


async def cli_run_async() -> None:
    """Run game server."""
    host = await find_ip()
    port = DEFAULT_PORT
    await run_server(GameServer, host, port)


def cli_run() -> None:
    """Run game server."""
    trio.run(cli_run_async)


if __name__ == "__main__":
    cli_run()
