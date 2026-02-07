#!/usr/bin/env python3
# AI that plays azul.

"""Minimax Azul AI."""

from __future__ import annotations

# Programmed by CoolCat467

__title__ = "Minimax AI"
__author__ = "CoolCat467"
__version__ = "0.0.0"

import time
from collections.abc import Hashable, Iterable, Mapping
from enum import IntEnum, auto
from math import inf as infinity
from typing import Any, ClassVar, Self, TypeAlias, TypeVar

from azul.state import (
    Phase,
    SelectableDestinationTiles,
    SelectableSourceTiles,
    State,
)
from azul_computer_players.machine_client import (
    RemoteState,
    run_clients_in_local_servers_sync,
)
from azul_computer_players.minimax import Minimax, MinimaxResult, Player

T = TypeVar("T")
Action: TypeAlias = (
    tuple[SelectableDestinationTiles, ...]
    | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]]
)


class TranspositionFlag(IntEnum):
    """Flag enum for transposition table."""

    LOWERBOUND = 0
    EXACT = auto()
    UPPERBOUND = auto()


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


class MinimaxWithID(Minimax[State, Action]):
    """Minimax with ID."""

    __slots__ = ()

    # Simple Transposition Table:
    # key -> (stored_depth, value, action, flag)
    # flag: TranspositionFlag: EXACT, LOWERBOUND, UPPERBOUND
    TRANSPOSITION_TABLE: ClassVar[
        dict[int, tuple[int, MinimaxResult[Any], TranspositionFlag]]
    ] = {}

    @classmethod
    def _transposition_table_lookup(
        cls,
        state_hash: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> MinimaxResult[Action] | None:
        """Lookup in transposition_table.  Return (value, action) or None."""
        entry = cls.TRANSPOSITION_TABLE.get(state_hash)
        if entry is None:
            return None

        stored_depth, result, flag = entry
        # only use if stored depth is deep enough
        if stored_depth >= depth and (
            (flag == TranspositionFlag.EXACT)
            or (flag == TranspositionFlag.LOWERBOUND and result.value > alpha)
            or (flag == TranspositionFlag.UPPERBOUND and result.value < beta)
        ):
            return result
        return None

    @classmethod
    def _transposition_table_store(
        cls,
        state_hash: int,
        depth: int,
        result: MinimaxResult[Action],
        alpha: float,
        beta: float,
    ) -> None:
        """Store in transposition_table with proper flag."""
        if result.value <= alpha:
            flag = TranspositionFlag.UPPERBOUND
        elif result.value >= beta:
            flag = TranspositionFlag.LOWERBOUND
        else:
            flag = TranspositionFlag.EXACT
        cls.TRANSPOSITION_TABLE[state_hash] = (depth, result, flag)

    @classmethod
    def hash_state(cls, state: State) -> int:
        """Your state-to-hash function.  Must be consistent."""
        # For small games you might do: return hash(state)
        # For larger, use Zobrist or custom.
        return hash(state)

    @classmethod
    def alphabeta_transposition_table(
        cls,
        state: State,
        depth: int = 5,
        a: int | float = -infinity,
        b: int | float = infinity,
    ) -> MinimaxResult[Action]:
        """AlphaBeta with transposition table."""
        if cls.terminal(state):
            return MinimaxResult(cls.value(state), None)
        if depth <= 0:
            ##            # Choose a random action
            ##            # No need for cryptographic secure random
            return MinimaxResult(
                cls.value(state),
                next(iter(cls.actions(state))),
            )
        next_down = depth - 1

        state_h = cls.hash_state(state)
        # 1) Try transposition_table lookup
        transposition_table_hit = cls._transposition_table_lookup(
            state_h,
            depth,
            a,
            b,
        )
        if transposition_table_hit is not None:
            return transposition_table_hit
        next_down = None if depth is None else depth - 1

        current_player = cls.player(state)
        value: int | float

        best_action: Action | None = None

        if current_player == Player.MAX:
            value = -infinity
            actions: list[tuple[Action, State]] = [
                (action, cls.result(state, action))
                for action in cls.actions(state)
            ]

            actions.sort(key=lambda act: cls.value(act[1]), reverse=True)
            for action, next_state in actions:
                child = cls.alphabeta_transposition_table(
                    next_state,
                    next_down,
                    a,
                    b,
                )
                if child.value > value:
                    value = child.value
                    best_action = action
                a = max(a, value)
                if a >= b:
                    break

        elif current_player == Player.MIN:
            value = infinity
            actions = [
                (action, cls.result(state, action))
                for action in cls.actions(state)
            ]

            actions.sort(key=lambda act: cls.value(act[1]))
            for action, next_state in actions:
                child = cls.alphabeta_transposition_table(
                    next_state,
                    next_down,
                    a,
                    b,
                )
                if child.value < value:
                    value = child.value
                    best_action = action
                b = min(b, value)
                if b <= a:
                    break
        else:
            raise NotImplementedError(f"{current_player = }")

        # 2) Store in transposition_table
        result = MinimaxResult(value, best_action)
        cls._transposition_table_store(
            state_h,
            depth,
            result,  # type: ignore[arg-type]
            a,
            b,
        )
        return result  # type: ignore[return-value]

    @classmethod
    def iterative_deepening(
        cls,
        state: State,
        start_depth: int = 5,
        max_depth: int = 7,
        time_limit_ns: int | float | None = None,
    ) -> MinimaxResult[Action]:
        """Run alpha-beta with increasing depth up to max_depth.

        If time_limit_ns is None, do all depths. Otherwise stop early.
        """
        best_result: MinimaxResult[Action] = MinimaxResult(0, None)
        start_t = time.perf_counter_ns()

        for depth in range(start_depth, max_depth + 1):
            # clear or keep transposition_table between depths? often you keep it
            # cls.TRANSPOSITION_TABLE.clear()

            result = cls.alphabeta_transposition_table(
                state,
                depth,
            )
            best_result = result

            if abs(result.value) == cls.HIGHEST:
                print(f"reached terminal state stop {depth=}")
                break

            # optional time check
            if (
                time_limit_ns
                and (time.perf_counter_ns() - start_t) > time_limit_ns
            ):
                print(
                    f"break from time expired {depth=} ({(time.perf_counter_ns() - start_t) / 1e9} seconds elaped)",
                )
                break
            print(
                f"{depth=} ({(time.perf_counter_ns() - start_t) / 1e9} seconds elaped)",
            )

        return best_result


MAX_PLAYER = 0


def convert_hashable(obj: object) -> Hashable:
    """Convert object to hashable object."""
    exc: TypeError | None = None
    try:
        hash(obj)
    except TypeError as exc:  # noqa: F841
        pass
    else:
        return obj
    if isinstance(obj, Mapping):
        return tuple(map(convert_hashable, obj.items()))
    if isinstance(obj, Iterable):
        return tuple(map(convert_hashable, obj))
    if exc is not None:
        raise NotImplementedError(type(obj)) from exc
    raise NotImplementedError(type(obj))


# Minimax[tuple[State, u8], Action]
class AzulMinimax(MinimaxWithID):
    """Minimax Algorithm for Azul."""

    __slots__ = ()

    @classmethod
    def hash_state(cls, state: AutoWallState) -> int:  # type: ignore[override]
        """Return state hash value."""
        # For small games you might do: return hash(state)
        # For larger, use Zobrist or custom.
        ##        return hash((state.size, tuple(state.pieces.items()), state.turn))
        return hash(convert_hashable(state))

    @staticmethod
    def value(state: State) -> int | float:
        """Return value of given game state."""
        # Real
        if AzulMinimax.terminal(state):
            winner, _score = state.get_win_order()[0]
            if winner == MAX_PLAYER:
                return 10
            return -10
        # Heuristic
        min_ = 0
        max_ = 0
        for player_id, player_data in state.player_data.items():
            score = player_data.get_end_of_game_score()
            score += player_data.get_floor_line_scoring()
            if player_id == MAX_PLAYER:
                max_ += score
            else:
                min_ += score
        # More max will make score higher,
        # more min will make score lower
        # Plus one in divisor makes so never / 0
        return (max_ - min_) / (abs(max_) + abs(min_) + 1)

    @staticmethod
    def terminal(state: State) -> bool:
        """Return if game state is terminal."""
        return state.current_phase == Phase.end

    @staticmethod
    def player(state: State) -> Player:
        """Return Player enum from current state's turn."""
        return Player.MAX if state.current_turn == MAX_PLAYER else Player.MIN

    @staticmethod
    def actions(state: State) -> Iterable[Action]:
        """Return all actions that are able to be performed for the current player in the given state."""
        return tuple(state.yield_actions())
        ##        print(f'{len(actions) = }')

    @staticmethod
    def result(
        state: State,
        action: Action,
    ) -> State:
        """Return new state after performing given action on given current state."""
        ##        real_state, MAX_PLAYER = state
        ##        return (real_state.preform_action(action), MAX_PLAYER)
        return state.preform_action(action)


##    @classmethod
##    def alphabeta(
##        cls,
##        state: tuple[State, u8],
##        depth: int | None = 5,
##        a: int | float = -infinity,
##        b: int | float = infinity,
##    ) -> MinimaxResult[
##        tuple[SelectableDestinationTiles, ...]
##        | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]]
##    ]:
##        """Return minimax alphabeta pruning result best action for given current state."""
##        new_state, player = state
##        if (
##            new_state.current_phase == Phase.wall_tiling
##            and not new_state.variant_play
##        ):
##            new_state = new_state.apply_auto_wall_tiling()
##        return super().alphabeta((new_state, player), depth, a, b)


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
        if not isinstance(self.state, AutoWallState):
            self.state = AutoWallState._make(self.state)
        assert isinstance(self.state, AutoWallState)
        ##        value, action = AzulMinimax.alphabeta((self.state, self.playing_as), 2)
        ##        value, action = AzulMinimax.alphabeta((self.state, self.playing_as), 4)
        global MAX_PLAYER
        MAX_PLAYER = self.playing_as
        value, action = AzulMinimax.iterative_deepening(
            self.state,
            2,
            20,
            int(5 * 1e9),
        )
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
