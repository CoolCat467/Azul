#!/usr/bin/env python3
# Azul Game Server

"""Azul Game Server."""

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

import traceback
from collections import deque
from enum import IntEnum, auto
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
    decode_cursor_location,
    encode_cursor_location,
    encode_int8_array,
    encode_numeric_uint8_counter,
    encode_tile_count,
)
from azul.state import Phase, State, Tile

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Awaitable, Callable

    from numpy import int8
    from numpy.typing import NDArray


# cursor_set_movement_mode
# cursor_set_destination


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
                "server[write]->playing_as": cbe.playing_as,
                "server[write]->game_over": cbe.game_over,
                "server[write]->board_data": cbe.board_data,
                "server[write]->pattern_data": cbe.pattern_data,
                "server[write]->factory_data": cbe.factory_data,
                "server[write]->cursor_data": cbe.cursor_data,
                "server[write]->table_data": cbe.table_data,
                "server[write]->cursor_movement_mode": cbe.cursor_movement_mode,
                "server[write]->current_turn_change": cbe.current_turn_change,
                "server[write]->cursor_position": cbe.cursor_position,
            },
        )
        sbe = ServerBoundEvents
        self.register_read_network_events(
            {
                sbe.encryption_response: f"client[{self.client_id}]->encryption_response",
                sbe.factory_clicked: f"client[{self.client_id}]->factory_clicked",
                sbe.cursor_location: f"client[{self.client_id}]->cursor_location",
                sbe.pattern_row_clicked: f"client[{self.client_id}]->pattern_row_clicked",
            },
        )

    def bind_handlers(self) -> None:
        """Bind event handlers."""
        super().bind_handlers()
        self.register_handlers(
            {
                f"client[{self.client_id}]->encryption_response": self.handle_encryption_response,
                f"client[{self.client_id}]->factory_clicked": self.read_factory_clicked,
                f"client[{self.client_id}]->cursor_location": self.read_cursor_location,
                f"client[{self.client_id}]->pattern_row_clicked": self.read_pattern_row_clicked,
                f"callback_ping->network[{self.client_id}]": self.handle_callback_ping,
                "initial_config->network": self.write_factory_clicked,
                f"playing_as->network[{self.client_id}]": self.write_playing_as,
                "game_over->network": self.write_game_over,
                "board_data->network": self.write_board_data,
                "factory_data->network": self.write_factory_data,
                "cursor_data->network": self.write_cursor_data,
                "table_data->network": self.write_table_data,
                f"cursor_movement_mode->network[{self.client_id}]": self.write_cursor_movement_mode,
                f"cursor_position->network[{self.client_id}]": self.write_cursor_position,
                "current_turn_change->network": self.write_current_turn_change,
                "pattern_data->network": self.write_pattern_data,
            },
        )

    async def start_encryption_request(self) -> None:
        """Start encryption request and raise as `server[write]->encryption_request`."""
        await super().start_encryption_request()

        event = await self.read_event()
        if event.name != f"client[{self.client_id}]->encryption_response":
            raise RuntimeError(
                f"Expected encryption response, got but {event.name!r}",
            )
        await self.handle_encryption_response(event)

    async def read_factory_clicked(self, event: Event[bytearray]) -> None:
        """Read factory_clicked event from client. Raise as `factory_clicked->server`."""
        buffer = Buffer(event.data)

        factory_id = buffer.read_value(StructFormat.UBYTE)
        tile_color = Tile(buffer.read_value(StructFormat.UBYTE))

        await self.raise_event(
            Event(
                "factory_clicked->server",
                (
                    self.client_id,
                    factory_id,
                    tile_color,
                ),
            ),
        )

    async def read_cursor_location(self, event: Event[bytearray]) -> None:
        """Read factory_clicked event from client. Raise as `factory_clicked->server`."""
        x, y = decode_cursor_location(event.data)

        await self.raise_event(
            Event(
                "cursor_location->server",
                (
                    self.client_id,
                    (x, y),
                ),
            ),
        )

    async def read_pattern_row_clicked(self, event: Event[bytearray]) -> None:
        """Read pattern_row_clicked event from client. Raise as `pattern_row_clicked->server`."""
        buffer = Buffer(event.data)

        row_id = buffer.read_value(StructFormat.UBYTE)
        row_pos_x = buffer.read_value(StructFormat.UBYTE)
        row_pos_y = buffer.read_value(StructFormat.UBYTE)

        await self.raise_event(
            Event(
                "pattern_row_clicked->server",
                (
                    self.client_id,
                    row_id,
                    (row_pos_x, row_pos_y),
                ),
            ),
        )

    async def handle_callback_ping(
        self,
        _: Event[None],
    ) -> None:
        """Reraise as server[write]->callback_ping."""
        await self.write_callback_ping()

    async def write_factory_clicked(
        self,
        event: Event[tuple[bool, int, int, int]],
    ) -> None:
        """Read initial config event and reraise as server[write]->initial_config."""
        varient_play, player_count, factory_count, current_turn = event.data

        buffer = Buffer()

        buffer.write_value(StructFormat.BOOL, varient_play)
        buffer.write_value(StructFormat.UBYTE, player_count)
        buffer.write_value(StructFormat.UBYTE, factory_count)
        buffer.write_value(StructFormat.UBYTE, current_turn)

        await self.write_event(Event("server[write]->initial_config", buffer))

    async def write_playing_as(
        self,
        event: Event[int],
    ) -> None:
        """Read playing as event and reraise as server[write]->playing_as."""
        playing_as = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.UBYTE, playing_as)
        await self.write_event(Event("server[write]->playing_as", buffer))

    async def write_game_over(self, event: Event[int]) -> None:
        """Read game over event and reraise as server[write]->game_over."""
        winner = event.data

        buffer = Buffer()

        buffer.write_value(StructFormat.UBYTE, winner)

        await self.write_event(Event("server[write]->game_over", buffer))

    async def write_board_data(
        self,
        event: Event[tuple[int, NDArray[int8]]],
    ) -> None:
        """Reraise as server[write]->board_data."""
        player_id, array = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.UBYTE, player_id)
        buffer.extend(encode_int8_array(array))

        await self.write_event(Event("server[write]->board_data", buffer))

    async def write_factory_data(
        self,
        event: Event[tuple[int, Counter[int]]],
    ) -> None:
        """Reraise as server[write]->factory_data."""
        factory_id, tiles = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.UBYTE, factory_id)
        buffer.extend(encode_numeric_uint8_counter(tiles))

        await self.write_event(Event("server[write]->factory_data", buffer))

    async def write_cursor_data(
        self,
        event: Event[Counter[int]],
    ) -> None:
        """Reraise as server[write]->cursor_data."""
        tiles = event.data

        buffer = encode_numeric_uint8_counter(tiles)

        await self.write_event(Event("server[write]->cursor_data", buffer))

    async def write_table_data(
        self,
        event: Event[Counter[int]],
    ) -> None:
        """Reraise as server[write]->table_data."""
        tiles = event.data

        buffer = encode_numeric_uint8_counter(tiles)

        await self.write_event(Event("server[write]->table_data", buffer))

    async def write_cursor_movement_mode(
        self,
        event: Event[bool],
    ) -> None:
        """Reraise as server[write]->cursor_movement_mode."""
        client_mode = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.BOOL, client_mode)

        await self.write_event(
            Event("server[write]->cursor_movement_mode", buffer),
        )

    async def write_cursor_position(
        self,
        event: Event[tuple[int, int]],
    ) -> None:
        """Reraise as server[write]->cursor_position."""
        buffer = encode_cursor_location(event.data)

        await self.write_event(
            Event("server[write]->cursor_position", buffer),
        )

    async def write_current_turn_change(
        self,
        event: Event[int],
    ) -> None:
        """Reraise as server[write]->current_turn_change."""
        pattern_id = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.UBYTE, pattern_id)

        await self.write_event(
            Event("server[write]->current_turn_change", buffer),
        )

    async def write_pattern_data(
        self,
        event: Event[tuple[int, int, tuple[int, int]]],
    ) -> None:
        """Reraise as server[write]->board_data."""
        player_id, row_id, (tile_color, tile_count) = event.data

        buffer = Buffer()
        buffer.write_value(StructFormat.UBYTE, player_id)
        buffer.write_value(StructFormat.UBYTE, row_id)
        assert tile_color >= 0
        buffer.extend(encode_tile_count(tile_color, tile_count))

        await self.write_event(
            Event("server[write]->pattern_data", buffer),
        )


class ServerPlayer(IntEnum):
    """Server Player enum."""

    one = 0
    two = auto()
    three = auto()
    four = auto()
    singleplayer_all = auto()
    spectator = auto()


class GameServer(network.Server):
    """Azul server.

    Handles accepting incoming connections from clients and handles
    main game logic via State subclass above.
    """

    __slots__ = (
        "actions_queue",
        "advertisement_scope",
        "client_count",
        "client_players",
        "internal_singleplayer_mode",
        "players_can_interact",
        "running",
        "state",
    )

    max_clients = 4

    def __init__(self, internal_singleplayer_mode: bool = False) -> None:
        """Initialize server."""
        super().__init__("GameServer")

        self.client_count: int = 0
        self.state = State.blank()

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
                "factory_clicked->server": self.handle_client_factory_clicked,
                "pattern_row_clicked->server": self.handle_client_pattern_row_clicked,
                "cursor_location->server": self.handle_cursor_location,
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
                players[client_id] = ServerPlayer.singleplayer_all
            else:
                players[client_id] = ServerPlayer.spectator
        return players

    @staticmethod
    def setup_teams(client_ids: list[int]) -> dict[int, int]:
        """Return teams given sorted client ids."""
        players: dict[int, int] = {}
        for idx, client_id in enumerate(client_ids):
            if idx < 4:
                players[client_id] = ServerPlayer(idx % 4)
            else:
                players[client_id] = ServerPlayer.spectator
        return players

    def new_game_init(self, varient_play: bool = False) -> None:
        """Start new game."""
        print("server new_game_init")
        self.client_players.clear()

        self.state = State.new_game(
            max(2, min(4, self.client_count)),
            varient_play,
        )

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

    async def handle_server_start_new_game(self, event: Event[bool]) -> None:
        """Handle game start."""
        varient_play = event.data
        ##        # Delete all pieces from last state (shouldn't be needed but still.)
        ##        async with trio.open_nursery() as nursery:
        ##            for piece_pos, _piece_type in self.state.get_pieces():
        ##                nursery.start_soon(
        ##                    self.raise_event,
        ##                    Event("delete_piece->network", piece_pos),
        ##                )

        # Choose which team plays first
        # Using non-cryptographically secure random because it doesn't matter
        self.new_game_init(varient_play)

        ##        # Send create_piece events for all pieces
        ##        async with trio.open_nursery() as nursery:
        ##            for piece_pos, piece_type in self.state.get_pieces():
        ##                nursery.start_soon(
        ##                    self.raise_event,
        ##                    Event("create_piece->network", (piece_pos, piece_type)),
        ##                )

        # Raise initial config event with board size and initial turn.
        await self.raise_event(
            Event(
                "initial_config->network",
                (
                    self.state.varient_play,
                    len(self.state.player_data),
                    len(self.state.factory_displays),
                    self.state.current_turn,
                ),
            ),
        )

        async with trio.open_nursery() as nursery:
            # Transmit board data
            for player_id, player_data in self.state.player_data.items():
                nursery.start_soon(
                    self.raise_event,
                    Event(
                        "board_data->network",
                        (
                            player_id,
                            player_data.wall,
                        ),
                    ),
                )
            # Transmit factory data
            for (
                factory_id,
                factory_tiles,
            ) in self.state.factory_displays.items():
                nursery.start_soon(
                    self.raise_event,
                    Event(
                        "factory_data->network",
                        (
                            factory_id,
                            factory_tiles,
                        ),
                    ),
                )
        # Transmit table center data
        await self.raise_event(
            Event(
                "table_data->network",
                self.state.table_center,
            ),
        )

        await self.transmit_cursor_movement_mode()

        await self.transmit_playing_as()

    async def transmit_cursor_movement_mode(self) -> None:
        """Update current cursor movement mode for all clients."""
        client_id = self.find_client_id_from_state_turn(
            self.state.current_turn,
        )

        await self.raise_event(
            Event(
                f"cursor_movement_mode->network[{client_id}]",
                True,
            ),
        )

        async with trio.open_nursery() as nursery:
            for other_client_id in self.client_players:
                if other_client_id != client_id:
                    nursery.start_soon(
                        self.raise_event,
                        Event(
                            f"cursor_movement_mode->network[{other_client_id}]",
                            False,
                        ),
                    )

    async def client_network_loop(
        self,
        client: ServerClient,
        controls_lobby: bool = False,
    ) -> None:
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
                # if controls_lobby:
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
        return self.state.current_phase != Phase.end

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
                        (
                            self.state.varient_play,
                            len(self.state.player_data),
                            len(self.state.factory_displays),
                            self.state.current_turn,
                        ),
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
            self.new_game_init()
        new_client_id = self.client_count

        # Is controlling player?
        is_zee_capitan = new_client_id == 0

        print(
            f"{self.__class__.__name__}: client connected [client_id {new_client_id}]",
        )
        self.client_count += 1

        can_start = self.can_start()
        print(f"[azul.server] {can_start = }")
        game_active = self.game_active()
        print(f"[azul.server] {game_active = }")
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
                print("TODO: Joined as spectator")
                # await self.send_spectator_join_packets(client)
            with self.temporary_component(client):
                if can_start and not game_active:  # and is_zee_capitan:
                    print("[azul.server] game start trigger.")
                    varient_play = False
                    await self.raise_event(
                        Event("server_send_game_start", varient_play),
                    )
                try:
                    await self.client_network_loop(client, is_zee_capitan)
                finally:
                    print(
                        f"{self.__class__.__name__}: client disconnected [client_id {new_client_id}]",
                    )
                    self.client_count -= 1
        # ServerClient's `with` block handles closing stream.

    def find_client_id_from_server_player_id(
        self,
        server_player_id: ServerPlayer,
    ) -> int | None:
        """Return client id from server player id or None if not found."""
        for client_id, current_server_player_id in self.client_players.items():
            if current_server_player_id == server_player_id:
                return client_id
            # Return singleplayer client id if exists
            if current_server_player_id == ServerPlayer.singleplayer_all:
                return client_id
        return None

    def find_server_player_id_from_state_turn(
        self,
        state_turn: int,
    ) -> ServerPlayer:
        """Return ServerPlayer id from game state turn."""
        if self.internal_singleplayer_mode:
            return ServerPlayer.singleplayer_all
        return ServerPlayer(state_turn)

    def find_client_id_from_state_turn(self, state_turn: int) -> int | None:
        """Return client id from state turn or None if not found."""
        server_player_id = self.find_server_player_id_from_state_turn(
            state_turn,
        )
        return self.find_client_id_from_server_player_id(server_player_id)

    async def handle_client_factory_clicked(
        self,
        event: Event[tuple[int, int, Tile]],
    ) -> None:
        """Handle client clicked a factory tile."""
        if not self.players_can_interact:
            print("Players are not allowed to interact.")
            await trio.lowlevel.checkpoint()
            return

        client_id, factory_id, tile = event.data

        server_player_id = self.client_players[client_id]

        if server_player_id == ServerPlayer.spectator:
            print(f"Spectator cannot select {factory_id = } {tile}")
            await trio.lowlevel.checkpoint()
            return

        player_id = int(server_player_id)
        if server_player_id == ServerPlayer.singleplayer_all:
            player_id = self.state.current_turn

        if player_id != self.state.current_turn:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select factory tile, not their turn.",
            )
            await trio.lowlevel.checkpoint()
            return

        if self.state.current_phase != Phase.factory_offer:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select factory tile, not in factory offer phase.",
            )
            await trio.lowlevel.checkpoint()
            return

        factory_display = self.state.factory_displays.get(factory_id)
        if factory_display is None:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select invalid factory {factory_id!r}.",
            )
            await trio.lowlevel.checkpoint()
            return

        if tile < 0 or tile not in factory_display:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select nonexistent color {tile}.",
            )
            await trio.lowlevel.checkpoint()
            return

        if not self.state.can_cursor_select_factory_color(
            factory_id,
            int(tile),
        ):
            print(
                f"Player {player_id} (client ID {client_id}) cannot select factory tile, state says no.",
            )
            await trio.lowlevel.checkpoint()
            return

        # Perform move
        self.state = self.state.cursor_selects_factory(factory_id, int(tile))

        # Send updates to client
        # Send factory display changes
        await self.raise_event(
            Event(
                "factory_data->network",
                (
                    factory_id,
                    self.state.factory_displays[factory_id],
                ),
            ),
        )
        await self.raise_event(
            Event(
                "cursor_data->network",
                self.state.cursor_contents,
            ),
        )
        await self.raise_event(
            Event(
                "table_data->network",
                self.state.table_center,
            ),
        )

    async def handle_client_pattern_row_clicked(
        self,
        event: Event[tuple[int, int, tuple[int, int]]],
    ) -> None:
        """Handle client clicking on pattern row."""
        if not self.players_can_interact:
            print("Players are not allowed to interact.")
            await trio.lowlevel.checkpoint()
            return

        client_id, row_id, row_pos = event.data

        server_player_id = self.client_players[client_id]

        if server_player_id == ServerPlayer.spectator:
            print(f"Spectator cannot select {row_id = } {row_pos}")
            await trio.lowlevel.checkpoint()
            return

        player_id = int(server_player_id)
        if server_player_id == ServerPlayer.singleplayer_all:
            player_id = self.state.current_turn

        if player_id != self.state.current_turn:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select pattern row, not their turn.",
            )
            await trio.lowlevel.checkpoint()
            return

        if self.state.current_phase != Phase.factory_offer:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select pattern row, not in factory offer phase.",
            )
            await trio.lowlevel.checkpoint()
            return

        if player_id != row_id:
            print(
                f"Player {player_id} (client ID {client_id}) cannot select pattern row {row_id} that does not belong to them.",
            )
            await trio.lowlevel.checkpoint()
            return

        column, line_id = row_pos
        place_count = 5 - column

        color = self.state.get_cursor_holding_color()

        max_place = self.state.get_player_line_max_placable_count(line_id)
        current_hold_count = self.state.cursor_contents[color]
        place_count = min(place_count, current_hold_count, max_place)

        if not self.state.can_player_select_line(line_id, color, place_count):
            print(
                f"Player {player_id} (client ID {client_id}) cannot select pattern line {line_id} placing {place_count} {Tile(color)} tiles.",
            )
            await trio.lowlevel.checkpoint()
            return

        prev_player_turn = self.state.current_turn

        self.state = self.state.player_selects_pattern_line(
            line_id,
            place_count,
        )

        if self.state.current_turn != player_id:
            if not self.internal_singleplayer_mode:
                new_client_id = self.find_client_id_from_state_turn(
                    self.state.current_turn,
                )
                assert new_client_id is not None
                await self.raise_event(
                    Event(
                        f"cursor_movement_mode->network[{client_id}]",
                        False,
                    ),
                )
                await self.raise_event(
                    Event(
                        f"cursor_movement_mode->network[{new_client_id}]",
                        True,
                    ),
                )

            await self.raise_event(
                Event(
                    "current_turn_change->network",
                    self.state.current_turn,
                ),
            )

        raw_tile_color, tile_count = self.state.player_data[
            prev_player_turn
        ].lines[line_id]
        # Do not send blank colors, clamp to zero
        tile_color = max(0, int(raw_tile_color))
        await self.raise_event(
            Event(
                "pattern_data->network",
                (
                    prev_player_turn,
                    line_id,
                    (tile_color, tile_count),
                ),
            ),
        )

        await self.raise_event(
            Event(
                "cursor_data->network",
                self.state.cursor_contents,
            ),
        )

    async def handle_cursor_location(
        self,
        event: Event[tuple[int, tuple[int, int]]],
    ) -> None:
        """Handle cursor location sent from client."""
        if not self.players_can_interact:
            print("Players are not allowed to interact.")
            await trio.lowlevel.checkpoint()
            return

        client_id, location = event.data

        server_player_id = self.client_players[client_id]

        if server_player_id == ServerPlayer.spectator:
            print("Spectator cannot control cursor")
            await trio.lowlevel.checkpoint()
            return

        player_id = int(server_player_id)
        if server_player_id == ServerPlayer.singleplayer_all:
            player_id = self.state.current_turn

        if player_id != self.state.current_turn:
            print(
                f"Player {player_id} (client ID {client_id}) cannot move cursor, not their turn.",
            )
            await trio.lowlevel.checkpoint()
            return

        # print(f"handle_cursor_location {client_id = } {location = }")

        if self.internal_singleplayer_mode:
            await trio.lowlevel.checkpoint()
            return

        async with trio.open_nursery() as nursery:
            for other_client_id in self.client_players:
                if other_client_id != client_id:
                    nursery.start_soon(
                        self.raise_event,
                        Event(
                            f"cursor_position->network[{other_client_id}]",
                            location,
                        ),
                    )

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
