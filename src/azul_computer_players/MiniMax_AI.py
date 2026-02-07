#!/usr/bin/env python3
# AI that plays checkers.

"""Minimax Checkers AI."""

from __future__ import annotations

# Programmed by CoolCat467

__title__ = "Minimax AI"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from math import inf as infinity
from typing import TYPE_CHECKING, TypeAlias, TypeVar

from machine_client import RemoteState, run_clients_in_local_servers_sync
from minimax import Minimax, MinimaxResult, Player
from mypy_extensions import u8

from azul.state import (
    Phase,
    SelectableDestinationTiles,
    SelectableSourceTiles,
    State,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

T = TypeVar("T")
Action: TypeAlias = (
    tuple[SelectableDestinationTiles, ...]
    | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]]
)

# Player:
# 0 = False = Person  = MIN = 0, 2
# 1 = True  = AI (Us) = MAX = 1, 3


class AutoWallState(State):
    """Azul State with automatic wall tiling in regular play mode."""

    __slots__ = ()

    def _factory_offer_maybe_next_turn(self) -> Self:
        """Return either current state or new state if player's turn is over."""
        new_state = super()._factory_offer_maybe_next_turn()

        if (
            new_state.current_phase == Phase.wall_tiling
            and not new_state.variant_play
        ):
            return new_state.apply_auto_wall_tiling()
        return new_state


class AzulMinimax(Minimax[tuple[AutoWallState, u8], Action]):
    """Minimax Algorithm for Checkers."""

    __slots__ = ()

    @staticmethod
    def value(state: tuple[AutoWallState, u8]) -> int | float:
        """Return value of given game state."""
        # Real
        real_state, max_player = state
        if AzulMinimax.terminal(state):
            winner, _score = real_state.get_win_order()[0]
            if winner == max_player:
                return 10
            return -10
        # Heuristic
        min_ = 0
        max_ = 0
        for player_id, player_data in real_state.player_data.items():
            score = player_data.get_end_of_game_score()
            score += player_data.get_floor_line_scoring()
            if player_id == max_player:
                max_ += score
            else:
                min_ += score
        # More max will make score higher,
        # more min will make score lower
        # Plus one in divisor makes so never / 0
        return (max_ - min_) / (abs(max_) + abs(min_) + 1)

    @staticmethod
    def terminal(state: tuple[AutoWallState, u8]) -> bool:
        """Return if game state is terminal."""
        real_state, _max_player = state
        return real_state.current_phase == Phase.end

    @staticmethod
    def player(state: tuple[AutoWallState, u8]) -> Player:
        """Return Player enum from current state's turn."""
        real_state, max_player = state
        return (
            Player.MAX if real_state.current_turn == max_player else Player.MIN
        )

    @staticmethod
    def actions(state: tuple[AutoWallState, u8]) -> Iterable[Action]:
        """Return all actions that are able to be performed for the current player in the given state."""
        real_state, _max_player = state
        return tuple(real_state.yield_actions())
        ##        print(f'{len(actions) = }')

    @staticmethod
    def result(
        state: tuple[AutoWallState, u8],
        action: Action,
    ) -> tuple[AutoWallState, u8]:
        """Return new state after performing given action on given current state."""
        real_state, max_player = state
        return (real_state.preform_action(action), max_player)

    @classmethod
    def adaptive_depth_minimax(
        cls,
        state: tuple[AutoWallState, u8],
    ) -> MinimaxResult[Action]:
        """Adaptive depth minimax."""
        # TODO
        depth = 1
        return cls.alphabeta(state, depth)

    @classmethod
    def alphabeta(
        cls,
        state: tuple[AutoWallState, u8],
        depth: int | None = 5,
        a: int | float = -infinity,
        b: int | float = infinity,
    ) -> MinimaxResult[
        tuple[SelectableDestinationTiles, ...]
        | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]]
    ]:
        """Return minimax alphabeta pruning result best action for given current state."""
        new_state, player = state
        if (
            new_state.current_phase == Phase.wall_tiling
            and not new_state.variant_play
        ):
            new_state = new_state.apply_auto_wall_tiling()
        return super().alphabeta((new_state, player), depth, a, b)


class MinimaxPlayer(RemoteState):
    """Minimax Player."""

    __slots__ = ()

    def __init__(self) -> None:
        """Initialize remote minmax player state."""
        super().__init__(state_class=AutoWallState)

    async def preform_turn(self) -> Action:
        """Perform turn."""
        print("preform_turn")
        ##value, action = CheckersMinimax.adaptive_depth_minimax(
        ##    self.state, 4, 5
        ##)
        ##value, action = CheckersMinimax.minimax(self.state, 4)
        assert isinstance(self.state, AutoWallState)
        value, action = AzulMinimax.alphabeta((self.state, self.playing_as), 2)
        ##        value, action = AzulMinimax.alphabeta((self.state, self.playing_as), 4)
        if action is None:
            raise ValueError("action is None")
        print(f"{value = }")
        return action


def run() -> None:
    """Run MinimaxPlayer clients in local server."""
    ##    import random
    ##
    ##    random.seed(0)
    ##
    ##    state = (AutoWallState.new_game(2), 0)
    ##
    ##    while not AzulMinimax.terminal(state):
    ##        action = AzulMinimax.adaptive_depth_minimax(state)
    ##        print(f"{action = }")
    ##        state = AzulMinimax.result(state, action.action)
    ##        print(f"{state = }")
    ##    print(state)

    run_clients_in_local_servers_sync(MinimaxPlayer)


if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.\n")
    run()
