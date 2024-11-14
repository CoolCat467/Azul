#!/usr/bin/env python3
# AI that plays checkers.

"""Minimax Checkers AI."""

from __future__ import annotations

# Programmed by CoolCat467

__title__ = "Minimax AI"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from typing import TYPE_CHECKING, TypeAlias, TypeVar

##from machine_client import RemoteState, run_clients_in_local_servers_sync
from minimax import Minimax, Player

from azul.state import (
    Phase,
    SelectableDestinationTiles,
    SelectableSourceTiles,
    State,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T")
Action: TypeAlias = (
    tuple[SelectableDestinationTiles, ...]
    | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]]
)

# Player:
# 0 = False = Person  = MIN = 0, 2
# 1 = True  = AI (Us) = MAX = 1, 3


class AzulMinimax(Minimax[State, Action]):
    """Minimax Algorithm for Checkers."""

    __slots__ = ()

    @staticmethod
    def value(state: State) -> int | float:
        """Return value of given game state."""
        # Real
        real_state, max_player = state
        if AzulMinimax.terminal(state):
            winner, _score = real_state.get_win_order()[0]
            if winner == max_player:
                return 1
            return -1
        # Heuristic
        min_ = 0
        max_ = 0
        for player_id, player_data in real_state.player_data.items():
            score = player_data.get_end_of_game_score()
            if player_id == max_player:
                max_ += score
            else:
                min_ += score
        # More max will make score higher,
        # more min will make score lower
        # Plus one in divisor makes so never / 0
        return (max_ - min_) / (max_ + min_ + 1)

    @staticmethod
    def terminal(state: State) -> bool:
        """Return if game state is terminal."""
        real_state, _max_player = state
        return real_state.current_phase == Phase.end

    @staticmethod
    def player(state: State) -> Player:
        """Return Player enum from current state's turn."""
        real_state, max_player = state
        return (
            Player.MAX if real_state.current_turn == max_player else Player.MIN
        )

    @staticmethod
    def actions(state: State) -> Iterable[Action]:
        """Return all actions that are able to be performed for the current player in the given state."""
        real_state, _max_player = state
        return tuple(real_state.yield_actions())
        ##        print(f'{len(actions) = }')

    @staticmethod
    def result(state: State, action: Action) -> State:
        """Return new state after performing given action on given current state."""
        real_state, max_player = state
        return (real_state.preform_action(action), max_player)


##class MinimaxPlayer(RemoteState):
##    """Minimax Player."""
##
##    __slots__ = ()
##
##    async def preform_turn(self) -> Action:
##        """Perform turn."""
##        print("preform_turn")
##        ##value, action = CheckersMinimax.adaptive_depth_minimax(
##        ##    self.state, 4, 5
##        ##)
##        ##value, action = CheckersMinimax.minimax(self.state, 4)
##        value, action = CheckersMinimax.alphabeta(self.state, 4)
##        if action is None:
##            raise ValueError("action is None")
##        print(f"{value = }")
##        return action


def run() -> None:
    """Run MinimaxPlayer clients in local server."""
    import random

    random.seed(0)

    state = (State.new_game(2), 0)

    while not AzulMinimax.terminal(state):
        action = AzulMinimax.alphabeta(state, 2)
        print(f"{action.value = }")
        state = AzulMinimax.result(state, action.action)
    print(state)


##    run_clients_in_local_servers_sync(MinimaxPlayer)


if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.\n")
    run()
