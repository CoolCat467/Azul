"""Machine Client - Checkers game client that can be controlled mechanically."""

from __future__ import annotations

__title__ = "Machine Client"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import sys
from abc import ABCMeta, abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeAlias, cast

import trio
from libcomponent.component import (
    Component,
    ComponentManager,
    Event,
    ExternalRaiseManager,
)

from azul.client import GameClient, read_advertisements
from azul.state import (
    PatternLine,
    Phase,
    SelectableDestination,
    SelectableDestinationTiles,
    SelectableSource,
    SelectableSourceTiles,
    State,
    Tile,
    factory_displays_deepcopy,
    player_data_deepcopy,
)
from azul.vector import Vector2

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import AsyncGenerator

    from mypy_extensions import u8
    from numpy import int8
    from numpy.typing import NDArray


if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

# Player:
# 0 = False = Person  = MIN = 0, 2
# 1 = True  = AI (Us) = MAX = 1, 3

##Action: TypeAlias = tuple[SelectableSourceTiles, SelectableDestinationTiles]
Action: TypeAlias = (
    tuple[SelectableDestinationTiles, ...]
    | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]]
)


class RemoteState(Component, metaclass=ABCMeta):
    """Remote State.

    Keeps track of game state and call preform_action when it's this clients
    turn.
    """

    __slots__ = (
        "can_made_play",
        "has_initial",
        "moves",
        "playing_as",
        "playing_lock",
        "state",
    )

    def __init__(self, state_class: type[State] = State) -> None:
        """Initialize remote state."""
        super().__init__("remote_state")

        ##        print(f'[RemoteState] {state_class = }')
        self.state = state_class.blank()
        self.has_initial = False

        self.playing_as: u8 = 1
        self.moves = 0

        self.playing_lock = trio.Lock()
        self.can_made_play = True

    def bind_handlers(self) -> None:
        """Register game event handlers."""
        self.register_handlers(
            {
                "game_winner": self.handle_game_over,
                "game_initial_config": self.handle_initial_config,
                "game_playing_as": self.handle_playing_as,
                "game_board_data": self.handle_board_data,
                "game_pattern_data": self.handle_pattern_data,
                "game_factory_data": self.handle_factory_data,
                "game_cursor_data": self.handle_cursor_data,
                "game_table_data": self.handle_table_data,
                # "game_cursor_set_movement_mode":
                "game_pattern_current_turn_change": self.handle_pattern_current_turn_change,
                # "game_cursor_set_destination":
                "game_floor_data": self.handle_floor_data,
            },
        )

    async def apply_select_source(
        self,
        selection: SelectableSourceTiles,
    ) -> None:
        """Select source."""
        ##        print(f"select {selection = }")
        color = selection.tiles
        if selection.source == SelectableSource.table_center:
            await self.raise_event(Event("game_table_clicked", color))
        elif selection.source == SelectableSource.factory:
            assert selection.source_id is not None
            await self.raise_event(
                Event("game_factory_clicked", (selection.source_id, color)),
            )
        else:
            raise NotImplementedError(selection.source)

    async def apply_select_destination(
        self,
        selection: SelectableDestinationTiles,
    ) -> None:
        """Select destination."""
        assert self.state.current_phase == Phase.factory_offer
        ##assert not self.state.is_cursor_empty()
        ##        print(f'dest {selection = }')

        if selection.destination == SelectableDestination.floor_line:
            await self.raise_event(
                Event(
                    "game_floor_clicked",
                    (self.playing_as, selection.place_count),
                ),
            )
        elif selection.destination == SelectableDestination.pattern_line:
            assert selection.destination_id is not None
            line_id = selection.destination_id
            currently_placed = self.state.get_player_line_current_place_count(
                line_id,
            )
            await self.raise_event(
                Event(
                    "game_pattern_row_clicked",
                    (
                        self.playing_as,
                        Vector2(
                            5 - selection.place_count - currently_placed,
                            line_id,
                        ),
                    ),
                ),
            )
        else:
            raise NotImplementedError(selection.destination)

    async def preform_action(self, action: Action) -> None:
        """Raise events to perform game action."""
        await self.raise_event(
            Event(
                "game_cursor_location_transmit",
                Vector2(0.5, 0.5),
            ),
        )
        source: SelectableSourceTiles | None = None
        dest: tuple[SelectableDestinationTiles, ...]
        if len(action) == 2:
            raw_source, raw_dest = action
            if isinstance(raw_source, SelectableSourceTiles):
                source = raw_source
                dest = cast("tuple[SelectableDestinationTiles, ...]", raw_dest)
            else:
                dest = cast("tuple[SelectableDestinationTiles, ...]", action)
        else:
            dest = action

        async with self.playing_lock:
            self.can_made_play = False
            if source is not None:
                await self.apply_select_source(source)
            for destination in dest:
                ##                print(f'{destination = }')
                assert isinstance(destination, SelectableDestinationTiles)
                await self.apply_select_destination(destination)
            self.can_made_play = True

    ##        raise NotImplementedError(f"{source = } {dest = }")

    @abstractmethod
    async def preform_turn(self) -> Action:
        """Perform turn, return action to perform."""

    async def base_preform_turn(self) -> None:
        """Perform turn."""
        ##        async with self.playing_lock:
        if not self.can_made_play:
            print("Skipping making move because of flag.")
            await trio.lowlevel.checkpoint()
            return
        self.can_made_play = False
        self.moves += 1
        ##        winner = self.state.check_for_win()
        ##        if winner is not None:
        if self.state.current_phase == Phase.end:
            print("Terminal state, not performing turn")
            ##value = ("Lost", "Won")[winner == self.playing_as]
            value = "<unknown>"
            print(f"{value} after {self.moves}")
            await trio.lowlevel.checkpoint()
            return
        print(f"\nMove {self.moves}...")
        action = await self.preform_turn()
        await self.preform_action(action)
        print("Action complete.")

    async def handle_playing_as(self, event: Event[u8]) -> None:
        """Handle client playing as specified player event."""
        print("handle_playing_as")
        self.playing_as = event.data

        if self.state.current_turn == self.playing_as:
            await self.base_preform_turn()
            return
        await trio.lowlevel.checkpoint()

    async def handle_initial_config(
        self,
        event: Event[tuple[u8, u8, u8, u8, NDArray[int8]]],
    ) -> None:
        """Set up initial game state."""
        ##        print("handle_initial_config")
        (
            variant_play,
            player_count,
            factory_count,
            current_turn,
            floor_line_data,
        ) = event.data
        ##        print(f'[RemoteState] {variant_play = }')
        self.state = State.new_game(player_count, bool(variant_play))
        self.state = self.state._replace(current_turn=current_turn)
        self.has_initial = True
        ##if current_turn == self.playing_as:
        ##    await self.base_preform_turn()

    async def handle_game_over(self, event: Event[u8]) -> None:
        """Raise network_stop event so we disconnect from server."""
        ##        print("handle_game_over")
        self.has_initial = False
        await self.raise_event(Event("network_stop", None))

    async def handle_board_data(
        self,
        event: Event[tuple[u8, NDArray[int8]]],
    ) -> None:
        """Handle player board data update."""
        ##        print("handle_board_data")
        player_id, board_data = event.data

        current_player_data = self.state.player_data[player_id]

        new_player_data = current_player_data._replace(wall=board_data)

        player_data = player_data_deepcopy(self.state.player_data)
        player_data[player_id] = new_player_data

        self.state = self.state._replace(
            player_data=player_data,
        )
        await trio.lowlevel.checkpoint()

    async def handle_pattern_data(
        self,
        event: Event[tuple[u8, u8, tuple[u8, u8]]],
    ) -> None:
        """Handle player pattern line data update."""
        ##        print("handle_pattern_data")
        player_id, row_id, (tile_color, tile_count) = event.data

        current_player_data = self.state.player_data[player_id]

        new_player_data = current_player_data._replace(
            lines=current_player_data.replace_pattern_line(
                current_player_data.lines,
                row_id,
                PatternLine(Tile(tile_color), int(tile_count)),
            ),
        )

        player_data = player_data_deepcopy(self.state.player_data)
        player_data[player_id] = new_player_data

        self.state = self.state._replace(
            player_data=player_data,
        )
        await trio.lowlevel.checkpoint()

    async def handle_factory_data(
        self,
        event: Event[tuple[u8, Counter[u8]]],
    ) -> None:
        """Handle factory data update."""
        ##        print("handle_factory_data")
        factory_id, tiles = event.data

        factory_displays = factory_displays_deepcopy(
            self.state.factory_displays,
        )
        factory_displays[factory_id] = tiles

        self.state = self.state._replace(
            factory_displays=factory_displays,
        )

        ##if self.state.current_turn == self.playing_as:
        ##    await self.base_preform_turn()
        ##    return
        await trio.lowlevel.checkpoint()

    async def handle_cursor_data(
        self,
        event: Event[Counter[u8]],
    ) -> None:
        """Handle cursor data update."""
        ##        print("handle_cursor_data")
        cursor_contents = event.data

        self.state = self.state._replace(
            cursor_contents=cursor_contents,
        )

        ##        if self.state.current_turn == self.playing_as and not self.state.is_cursor_empty():
        ##            await self.base_preform_turn()
        ##            return
        await trio.lowlevel.checkpoint()

    async def handle_table_data(self, event: Event[Counter[u8]]) -> None:
        """Handle table center tile data update."""
        ##        print("handle_table_data")
        table_center = event.data

        self.state = self.state._replace(
            table_center=table_center,
        )
        await trio.lowlevel.checkpoint()

    async def handle_pattern_current_turn_change(
        self,
        event: Event[u8],
    ) -> None:
        """Handle change of current turn."""
        print("handle_pattern_current_turn_change")
        pattern_id = event.data

        self.state = self.state._replace(
            current_turn=pattern_id,
        )

        if self.state.current_turn == self.playing_as:
            await self.base_preform_turn()
            return
        await trio.lowlevel.checkpoint()

    async def handle_floor_data(
        self,
        event: Event[tuple[u8, Counter[u8]]],
    ) -> None:
        """Handle floor data event."""
        ##        print("handle_floor_data")
        floor_id, floor_line = event.data

        current_player_data = self.state.player_data[floor_id]

        new_player_data = current_player_data._replace(floor=floor_line)

        player_data = player_data_deepcopy(self.state.player_data)
        player_data[floor_id] = new_player_data

        self.state = self.state._replace(
            player_data=player_data,
        )
        await trio.lowlevel.checkpoint()


class MachineClient(ComponentManager):
    """Manager that runs until client_disconnected event fires."""

    __slots__ = ("running",)

    def __init__(self, remote_state_class: type[RemoteState]) -> None:
        """Initialize machine client."""
        super().__init__("machine_client")

        self.running = True

        self.add_component(remote_state_class())

    @asynccontextmanager
    async def client_with_block(self) -> AsyncGenerator[GameClient, None]:
        """Add client temporarily with `with` block, ensuring closure."""
        async with GameClient("game_client") as client:
            with self.temporary_component(client):
                yield client

    def bind_handlers(self) -> None:
        """Register client event handlers."""
        self.register_handlers(
            {
                "client_disconnected": self.handle_client_disconnected,
                "client_connection_closed": self.handle_client_disconnected,
            },
        )

    ##    async def raise_event(self, event: Event) -> None:
    ##        """Raise event but also log it if not tick."""
    ##        if event.name not in {"tick"}:
    ##            print(f'{event = }')
    ##        return await super().raise_event(event)

    async def handle_client_disconnected(self, event: Event[None]) -> None:
        """Set self.running to false on network disconnect."""
        self.running = False


async def run_client(
    host: str,
    port: int,
    remote_state_class: type[RemoteState],
    connected: set[tuple[str, int]],
) -> None:
    """Run machine client and raise tick events."""
    async with trio.open_nursery() as main_nursery:
        event_manager = ExternalRaiseManager(
            "checkers",
            main_nursery,
            "client",
        )
        client = MachineClient(remote_state_class)
        with event_manager.temporary_component(client):
            async with client.client_with_block():
                await event_manager.raise_event(
                    Event("client_connect", (host, port)),
                )
                print(f"Connected to server {host}:{port}")
                try:
                    while client.running:  # noqa: ASYNC110
                        # Wait so backlog things happen
                        await trio.sleep(1)
                except KeyboardInterrupt:
                    print("Shutting down client from keyboard interrupt.")
                    await event_manager.raise_event(
                        Event("network_stop", None),
                    )
        print(f"Disconnected from server {host}:{port}")
        client.unbind_components()
    connected.remove((host, port))


def run_client_sync(
    host: str,
    port: int,
    remote_state_class: type[RemoteState],
) -> None:
    """Run client and connect to server at host:port."""
    trio.run(run_client, host, port, remote_state_class, set())


async def run_clients_in_local_servers(
    remote_state_class: type[RemoteState],
) -> None:
    """Run clients in local servers."""
    connected: set[tuple[str, int]] = set()
    print("Watching for advertisements...\n(CTRL + C to quit)")
    try:
        async with trio.open_nursery(strict_exception_groups=True) as nursery:
            while True:
                advertisements = set(await read_advertisements())
                servers = {server for _motd, server in advertisements}
                servers -= connected
                for server in servers:
                    connected.add(server)
                    nursery.start_soon(
                        run_client,
                        *server,
                        remote_state_class,
                        connected,
                    )
                await trio.sleep(1)
    except BaseExceptionGroup as exc:
        for ex in exc.exceptions:
            if isinstance(ex, KeyboardInterrupt):
                print("Shutting down from keyboard interrupt.")
                break
        else:
            raise


def run_clients_in_local_servers_sync(
    remote_state_class: type[RemoteState],
) -> None:
    """Run clients in local servers."""
    trio.run(run_clients_in_local_servers, remote_state_class)
