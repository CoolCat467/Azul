"""Azul State."""

# Programmed by CoolCat467

from __future__ import annotations

# Copyright (C) 2024  CoolCat467
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

__title__ = "Azul State"
__author__ = "CoolCat467"
__license__ = "GNU General Public License Version 3"
__version__ = "0.0.0"


import random
from collections import Counter
from enum import IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    NamedTuple,
    TypeVar,
)

from numpy import full, int8

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray
    from typing_extensions import Self

T = TypeVar("T")

FLOOR_LINE_COUNT: Final = 7


def floor_line_subtract_generator(seed: int = 1) -> Generator[int, None, None]:
    """Floor Line subtraction number generator. Can continue indefinitely."""
    while True:
        yield from (-seed,) * (seed + 1)
        seed += 1


FLOOR_LINE_DATA: Final = tuple(
    value
    for _, value in zip(
        range(FLOOR_LINE_COUNT),
        floor_line_subtract_generator(),
        strict=False,
    )
)


class Tile(IntEnum):
    """All type types."""

    blank = -6
    fake_cyan = -5
    fake_black = -4
    fake_red = -3
    fake_yellow = -2
    fake_blue = -1
    blue = 0
    yellow = auto()
    red = auto()
    black = auto()
    cyan = auto()
    one = auto()


REAL_TILES: Final = {Tile.blue, Tile.yellow, Tile.red, Tile.black, Tile.cyan}


class Phase(IntEnum):
    """Game phases."""

    factory_offer = 0
    wall_tiling = auto()
    end = auto()


def generate_bag_contents() -> Counter[int]:
    """Generate and return unrandomized bag."""
    tile_types = 5
    tile_count = 100
    count_each = tile_count // tile_types
    return Counter({type_: count_each for type_ in range(tile_types)})


def bag_draw_tile(bag: Counter[int]) -> int:
    """Return drawn tile from bag. Mutates bag."""
    # S311 Standard pseudo-random generators are not suitable for
    # cryptographic purposes
    tile = random.choice(tuple(bag.elements()))  # noqa: S311
    bag[tile] -= 1
    return tile


def select_color(holder: Counter[int], color: int) -> int:
    """Pop color tiles from bag. Returns count. Mutates holder.

    Raises KeyError if color not in holder.
    """
    return holder.pop(color)


class PatternLine(NamedTuple):
    """Player pattern line row."""

    color: Tile
    count_: int

    @classmethod
    def blank(cls) -> Self:
        """Return new blank pattern line."""
        return cls(
            color=Tile.blank,
            count_=0,
        )

    def place_tiles(self, color: Tile, place_count: int) -> Self:
        """Return new pattern line after placing <count> tiles of given color."""
        assert self.color == Tile.blank or self.color == color
        assert place_count > 0
        return self._replace(
            color=color,
            count_=self.count_ + place_count,
        )


def remove_counter_zeros(counter: Counter[Any]) -> None:
    """Remove any zero counts from given counter. Mutates counter."""
    for key, count in tuple(counter.items()):
        if count == 0:
            del counter[key]


def floor_fill_tile_excess(
    floor: Counter[int],
    tile: int,
    count: int,
) -> Counter[int]:
    """Fill floor with count of tile, return excess for box lid. Mutates floor."""
    excess: Counter[int] = Counter()
    while floor.total() < FLOOR_LINE_COUNT and count > 0:
        floor[tile] += 1
        count -= 1
    # If overflow and it's number one tile
    if count and tile == Tile.one:
        # Move non-one tiles from floor to excess
        non_one = floor.total() - floor[Tile.one]
        assert non_one > 0
        for _ in range(min(non_one, count)):
            non_one_tiles = set(floor.elements()) - {Tile.one}
            non_one_tile = sorted(non_one_tiles).pop()
            # Move non-one tile from floor to box lid
            floor[non_one_tile] -= 1
            excess[non_one_tile] += 1
            # Add one tile to floor
            floor[tile] += 1
            count -= 1
        remove_counter_zeros(floor)
    assert count >= 0
    if count:
        # Add overflow tiles to box lid.
        excess[tile] += count

    return excess


class UnplayableTileError(Exception):
    """Unplayable Tile Exception."""

    __slots__ = ("y",)

    def __init__(self, y: int) -> None:
        """Remember Y position."""
        self.y = y


class PlayerData(NamedTuple):
    """Player data."""

    score: int
    wall: NDArray[int8]
    lines: tuple[PatternLine, ...]
    floor: Counter[int]

    @classmethod
    def new(cls, variant_play: bool = False) -> Self:
        """Return new player data instance."""
        wall = full((5, 5), Tile.blank, int8)

        if not variant_play:
            for y in range(5):
                for x in range(5):
                    color = -((5 - y + x) % len(REAL_TILES) + 1)
                    wall[y, x] = color

        return cls(
            score=0,
            wall=wall,
            lines=(PatternLine.blank(),) * 5,
            floor=Counter(),
        )

    def copy(self) -> Self:
        """Return copy of self."""
        return self._replace(
            floor=self.floor.copy(),
        )

    def line_id_valid(self, line_id: int) -> bool:
        """Return if given line id is valid."""
        return line_id >= 0 and line_id < len(self.lines)

    @staticmethod
    def get_line_max_count(line_id: int) -> int:
        """Return max count allowed in given line."""
        # Line id is keeping track of max count
        return line_id + 1

    def get_line_current_place_count(self, line_id: int) -> int:
        """Return count of currently placed tiles for given line."""
        assert self.line_id_valid(line_id)
        return self.lines[line_id].count_

    def get_line_max_placable_count(self, line_id: int) -> int:
        """Return max placable count for given line."""
        assert self.line_id_valid(line_id)
        max_count = self.get_line_max_count(line_id)
        return max_count - self.lines[line_id].count_

    def get_row_colors_used(self, line_id: int) -> set[Tile]:
        """Return set of tile colors used in wall for given row."""
        row = self.wall[line_id, :]
        return {Tile(int(x)) for x in row[row >= 0]}

    def get_row_unused_colors(self, line_id: int) -> set[Tile]:
        """Return set of tiles colors not currently used in wall for given row."""
        return REAL_TILES - self.get_row_colors_used(line_id)

    def yield_possible_placement_rows(
        self,
        color: int,
    ) -> Generator[tuple[int, int], None, None]:
        """Yield row line ids and number of placable for rows able to place color at."""
        for line_id, line in enumerate(self.lines):
            # Color must match
            if line.color != Tile.blank and int(line.color) != color:
                # print("color mismatch")
                continue
            placable = self.get_line_max_placable_count(line_id)
            # Must have placable spots
            if not placable:
                continue
            # Must not already use color
            if color in self.get_row_colors_used(line_id):
                continue
            yield (line_id, placable)

    def can_select_line(
        self,
        line_id: int,
        color: int,
        place_count: int,
    ) -> bool:
        """Return if can select given line with given color and place count."""
        if not self.line_id_valid(line_id):
            # print("invalid line id")
            return False
        line = self.lines[line_id]
        # Don't allow placing zero
        if place_count <= 0:
            # print("place count too smol")
            return False
        # Color must match
        if line.color != Tile.blank and int(line.color) != color:
            # print("color mismatch")
            return False
        # Must have space to place
        if place_count > self.get_line_max_placable_count(line_id):
            return False
        # Can't place in row that uses that color already
        return Tile(color) not in self.get_row_colors_used(line_id)

    @staticmethod
    def replace_pattern_line(
        lines: tuple[PatternLine, ...],
        line_id: int,
        new: PatternLine,
    ) -> tuple[PatternLine, ...]:
        """Return new pattern line data after replacing one of them."""
        left = lines[:line_id]
        right = lines[line_id + 1 :]
        return (*left, new, *right)

    def place_pattern_line_tiles(
        self,
        line_id: int,
        color: int,
        place_count: int,
    ) -> Self:
        """Return new player data after placing tiles in a pattern line."""
        assert self.can_select_line(line_id, color, place_count)
        line = self.lines[line_id]
        return self._replace(
            lines=self.replace_pattern_line(
                self.lines,
                line_id,
                line.place_tiles(Tile(color), place_count),
            ),
        )

    def is_floor_line_full(self) -> bool:
        """Return if floor line is full."""
        return self.floor.total() >= FLOOR_LINE_COUNT

    def place_floor_line_tiles(
        self,
        color: int,
        place_count: int,
    ) -> tuple[Self, Counter[int]]:
        """Return new player and excess tiles for box lid."""
        floor = self.floor.copy()
        for_box_lid = floor_fill_tile_excess(floor, color, place_count)
        assert all(x > 0 for x in for_box_lid.values()), for_box_lid
        return (
            self._replace(floor=floor),
            for_box_lid,
        )

    def get_horizontal_linked_wall_count(
        self,
        x: int,
        y: int,
        wall: NDArray[int8],
    ) -> int:
        """Return horizontally-linked tile count."""
        count = 0
        for range_ in (range(x - 1, -1, -1), range(x + 1, 5)):
            for cx in range_:
                if wall[y, cx] < 0:
                    break
                count += 1
        return count

    def get_vertically_linked_wall_count(
        self,
        x: int,
        y: int,
        wall: NDArray[int8],
    ) -> int:
        """Return vertically-linked tile count."""
        count = 0
        for range_ in (range(y - 1, -1, -1), range(y + 1, 5)):
            for cy in range_:
                if wall[cy, x] < 0:
                    break
                count += 1
        return count

    def get_score_from_wall_placement(
        self,
        color: int,
        x: int,
        y: int,
        wall: NDArray[int8],
    ) -> int:
        """Return score increment value from placing tile at given coordinates."""
        # Should be blank or fake at position
        assert wall[y, x] < 0
        count = 1
        count += self.get_horizontal_linked_wall_count(x, y, wall)
        count += self.get_vertically_linked_wall_count(x, y, wall)
        return count

    def get_floor_line_scoring(self) -> int:
        """Return score increment value from floor line."""
        total_count = self.floor.total()
        assert total_count <= FLOOR_LINE_COUNT
        score = 0
        for idx in range(total_count):
            score += FLOOR_LINE_DATA[idx]
        return score

    def perform_auto_wall_tiling(self) -> tuple[Self, Counter[int], bool]:
        """Return new player data and tiles for box lid after performing automatic wall tiling."""
        for_box_lid: Counter[int] = Counter()

        score = self.score
        new_lines = self.lines
        new_wall = self.wall.copy()
        for line_id, line in enumerate(self.lines):
            if line.count_ != self.get_line_max_count(line_id):
                continue
            left = max(0, line.count_ - 1)
            if left:
                for_box_lid[line.color] += left
            # placed tile is stuck in the wall now
            x = tuple(map(int, new_wall[line_id, :])).index(-line.color - 1)
            score += self.get_score_from_wall_placement(
                line.color,
                x,
                line_id,
                new_wall,
            )
            new_wall[line_id, x] = line.color
            new_lines = self.replace_pattern_line(
                new_lines,
                line_id,
                PatternLine.blank(),
            )

        score += self.get_floor_line_scoring()
        if score < 0:
            score = 0

        # Get one tile from floor line
        floor = self.floor.copy()
        has_one = False
        if floor[Tile.one]:
            floor[Tile.one] -= 1
            remove_counter_zeros(floor)
            has_one = True
        for_box_lid.update(floor)

        return (
            self._replace(
                lines=new_lines,
                wall=new_wall,
                score=score,
                floor=Counter(),
            ),
            for_box_lid,
            has_one,
        )

    def has_horizontal_wall_line(self) -> bool:
        """Return if full horizontal line is filled anywhere."""
        return any(all(self.wall[y, :] >= 0) for y in range(5))

    def get_filled_horizontal_line_count(self) -> int:
        """Return number of filled horizontal lines."""
        count = 0
        for y in range(5):
            if all(self.wall[y, :] >= 0):
                count += 1
        return count

    def get_end_of_game_score(self) -> int:
        """Return end of game score for this player."""
        score = self.score
        score += self.get_filled_horizontal_line_count() * 2
        for x in range(5):
            if all(self.wall[:, x] >= 0):
                score += 7
        counts = Counter(int(x) for x in self.wall[self.wall >= 0])
        for count in counts.values():
            if count == 5:
                score += 10
        return score

    def perform_end_of_game_scoring(self) -> Self:
        """Return new player data after performing end of game scoring."""
        return self._replace(score=self.get_end_of_game_score())

    def get_manual_wall_tile_location(self) -> tuple[int, list[int]] | None:
        """Return tuple of row and placable columns for wall tiling, or None if done.

        Raises UnplayableTileError if no valid placement locations.
        """
        for y, line in enumerate(self.lines):
            if line.color == Tile.blank:
                continue
            if line.count_ != self.get_line_max_count(y):
                continue

            valid_x: list[int] = []
            for x, is_open in enumerate(self.wall[y, :] >= 0):
                if not is_open:
                    continue
                if line.color in {Tile(int(v)) for v in self.wall[:, x]}:
                    continue
                valid_x.append(x)
            if not valid_x:
                raise UnplayableTileError(y)
            return (y, valid_x)
        return None

    def handle_unplayable_wall_tiling(
        self,
        y: int,
    ) -> tuple[Self, Counter[int]]:
        """Return new player data and tiles for floor line."""
        line = self.lines[y]
        assert line.color != Tile.blank

        new_lines = self.replace_pattern_line(
            self.lines,
            y,
            PatternLine.blank(),
        )

        return self._replace(
            lines=new_lines,
        ).place_floor_line_tiles(line.color, line.count_)

    def manual_wall_tiling_action(
        self,
        line_id: int,
        x_pos: int,
    ) -> tuple[Self, Counter[int]]:
        """Wall tile given full line to given x position in that row.

        Return new player data and any tiles to return to box lid.
        """
        for_box_lid: Counter[int] = Counter()

        score = self.score
        new_lines = self.lines
        new_wall = self.wall.copy()

        line = self.lines[line_id]

        assert line.count_ == self.get_line_max_count(line_id)
        assert line.color != Tile.blank
        assert new_wall[line_id, x_pos] == Tile.blank

        left = max(0, line.count_ - 1)
        if left:
            for_box_lid[line.color] += left
        # placed tile is stuck in wall now
        score += self.get_score_from_wall_placement(
            line.color,
            x_pos,
            line_id,
            new_wall,
        )
        new_wall[line_id, x_pos] = line.color
        new_lines = self.replace_pattern_line(
            new_lines,
            line_id,
            PatternLine.blank(),
        )

        return (
            self._replace(
                lines=new_lines,
                wall=new_wall,
                score=score,
            ),
            for_box_lid,
        )

    def finish_manual_wall_tiling(self) -> tuple[Self, Counter[int], bool]:
        """Return new player data and tiles for box lid after performing automatic wall tiling."""
        for_box_lid: Counter[int] = Counter()

        score = self.score

        score += self.get_floor_line_scoring()
        if score < 0:
            score = 0

        # Get one tile from floor line
        floor = self.floor.copy()
        has_one = False
        if floor[Tile.one]:
            floor[Tile.one] -= 1
            remove_counter_zeros(floor)
            has_one = True
        for_box_lid.update(floor)

        return (
            self._replace(
                score=score,
                floor=Counter(),
            ),
            for_box_lid,
            has_one,
        )


def factory_displays_deepcopy(
    factory_displays: dict[int, Counter[int]],
) -> dict[int, Counter[int]]:
    """Return deepcopy of factory displays."""
    return {k: v.copy() for k, v in factory_displays.items()}


def player_data_deepcopy(
    player_data: dict[int, PlayerData],
) -> dict[int, PlayerData]:
    """Return deepcopy of player data."""
    return {k: v.copy() for k, v in player_data.items()}


class SelectableSource(IntEnum):
    """Selectable tile source."""

    table_center = 0
    factory = auto()


class SelectableSourceTiles(NamedTuple):
    """Selectable source tiles data."""

    source: SelectableSource
    tiles: Tile
    # Factory ids
    source_id: int | None = None


class SelectableDestination(IntEnum):
    """Selectable tile destination."""

    floor_line = 0
    pattern_line = auto()


class SelectableDestinationTiles(NamedTuple):
    """Selectable destination tiles data."""

    destination: SelectableDestination
    place_count: int
    # Pattern line ids
    destination_id: int | None = None


class State(NamedTuple):
    """Represents state of an azul game."""

    variant_play: bool
    current_phase: Phase
    bag: Counter[int]
    box_lid: Counter[int]
    table_center: Counter[int]
    factory_displays: dict[int, Counter[int]]
    cursor_contents: Counter[int]
    current_turn: int
    player_data: dict[int, PlayerData]

    @classmethod
    def blank(cls) -> Self:
        """Return new blank state."""
        return cls(
            variant_play=False,
            current_phase=Phase.end,
            bag=Counter(),
            box_lid=Counter(),
            table_center=Counter(),
            factory_displays={},
            cursor_contents=Counter(),
            current_turn=0,
            player_data={},
        )

    @classmethod
    def new_game(cls, player_count: int, variant_play: bool = False) -> Self:
        """Return state of a new game."""
        factory_count = player_count * 2 + 1
        bag = generate_bag_contents()

        factory_displays: dict[int, Counter[int]] = {}
        for x in range(factory_count):
            tiles: Counter[int] = Counter()
            for _ in range(4):
                tiles[bag_draw_tile(bag)] += 1
            factory_displays[x] = tiles

        return cls(
            variant_play=variant_play,
            current_phase=Phase.factory_offer,
            bag=bag,
            box_lid=Counter(),
            table_center=Counter({Tile.one: 1}),
            factory_displays=factory_displays,
            cursor_contents=Counter(),
            current_turn=0,
            player_data={
                x: PlayerData.new(variant_play) for x in range(player_count)
            },
        )

    def is_cursor_empty(self) -> bool:
        """Return if cursor is empty."""
        return self.cursor_contents.total() == 0

    def can_cursor_select_factory(self, factory_id: int) -> bool:
        """Return if cursor can select a specific factory."""
        assert self.current_phase == Phase.factory_offer
        if not self.is_cursor_empty():
            return False
        factory = self.factory_displays.get(factory_id, None)
        if factory is None:
            return False
        return factory.total() > 0

    def can_cursor_select_factory_color(
        self,
        factory_id: int,
        color: int,
    ) -> bool:
        """Return if cursor can select color at factory."""
        if not self.can_cursor_select_factory(factory_id):
            return False
        factory = self.factory_displays[factory_id]
        return factory[color] > 0

    def cursor_selects_factory(self, factory_id: int, color: int) -> Self:
        """Return new state after cursor selects factory."""
        assert self.can_cursor_select_factory_color(factory_id, color)
        # Only mutate copies
        factory_displays = factory_displays_deepcopy(self.factory_displays)
        table_center = self.table_center.copy()
        cursor_contents = self.cursor_contents.copy()

        factory = factory_displays[factory_id]
        count = select_color(factory, color)
        # Add to cursor
        cursor_contents[color] += count
        # Add all non-matching colored tiles to center of table
        table_center.update(factory)
        factory.clear()

        return self._replace(
            table_center=table_center,
            factory_displays=factory_displays,
            cursor_contents=cursor_contents,
        )

    def can_cursor_select_center(self, color: int) -> bool:
        """Return if cursor can select color from table center."""
        assert self.current_phase == Phase.factory_offer
        if not self.is_cursor_empty():
            return False
        return color != Tile.one and self.table_center[color] > 0

    def cursor_selects_table_center(self, color: int) -> Self:
        """Return new state after cursor selects from table center."""
        assert self.can_cursor_select_center(color)
        table_center = self.table_center.copy()
        cursor_contents = self.cursor_contents.copy()

        # Get all of color from table center and add to cursor
        cursor_contents[color] += select_color(table_center, color)
        # Handle number one tile
        if table_center[Tile.one]:
            cursor_contents[Tile.one] += select_color(table_center, Tile.one)
        remove_counter_zeros(table_center)

        return self._replace(
            table_center=table_center,
            cursor_contents=cursor_contents,
        )

    def yield_table_center_selections(
        self,
    ) -> Generator[SelectableSourceTiles, None, None]:
        """Yield SelectableSourceTiles objects from table center."""
        for color, count in self.table_center.items():
            if color == Tile.one or count <= 0:
                continue
            yield SelectableSourceTiles(
                source=SelectableSource.table_center,
                tiles=Tile(color),
            )

    def yield_selectable_tiles_factory_offer(
        self,
    ) -> Generator[SelectableSourceTiles, None, None]:
        """Yield SelectableSourceTiles objects from all sources."""
        yield from self.yield_table_center_selections()
        for factory_id, factory_display in self.factory_displays.items():
            for color in factory_display:
                yield SelectableSourceTiles(
                    source=SelectableSource.factory,
                    tiles=Tile(color),
                    source_id=factory_id,
                )

    def apply_source_select_action_factory_offer(
        self,
        selection: SelectableSourceTiles,
    ) -> Self:
        """Return new state after applying selection action."""
        color = selection.tiles
        if selection.source == SelectableSource.table_center:
            return self.cursor_selects_table_center(color)
        if selection.source == SelectableSource.factory:
            assert selection.source_id is not None
            return self.cursor_selects_factory(selection.source_id, color)
        raise NotImplementedError(selection.source)

    def get_cursor_holding_color(self) -> int:
        """Return color of tile cursor is holding."""
        cursor_colors = set(self.cursor_contents.elements())
        # Do not count number one tile
        cursor_colors.discard(Tile.one)
        assert len(cursor_colors) == 1, "Cursor should only exactly one color"
        return cursor_colors.pop()

    def can_player_select_line(
        self,
        line_id: int,
        color: int,
        place_count: int,
    ) -> bool:
        """Return if player can select line."""
        player_data = self.player_data[self.current_turn]

        # Cannot place more than we have
        # Can't be pulling tiles out of thin air now can we?
        if place_count > self.cursor_contents[color]:
            return False

        return player_data.can_select_line(line_id, color, place_count)

    def get_player_line_max_placable_count(self, line_id: int) -> int:
        """Return max placable count for given line."""
        player_data = self.player_data[self.current_turn]

        return player_data.get_line_max_placable_count(line_id)

    def get_player_line_current_place_count(self, line_id: int) -> int:
        """Return current place count for given line."""
        player_data = self.player_data[self.current_turn]

        return player_data.get_line_current_place_count(line_id)

    def all_pullable_empty(self) -> bool:
        """Return if all pullable tile locations are empty, not counting cursor."""
        if self.table_center.total():
            return False
        for factory_display in self.factory_displays.values():
            if factory_display.total():
                return False
        return True

    def _factory_offer_maybe_next_turn(self) -> Self:
        """Return either current state or new state if player's turn is over."""
        assert self.current_phase == Phase.factory_offer
        # If cursor is still holding things, turn is not over.
        if not self.is_cursor_empty():
            return self
        # Turn is over
        # Increment who's turn it is
        current_turn = (self.current_turn + 1) % len(self.player_data)

        current_phase: Phase = self.current_phase
        if self.all_pullable_empty():
            # Go to wall tiling phase
            current_phase = Phase.wall_tiling

        ##if current_phase == Phase.wall_tiling and not self.variant_play:
        ##    return new_state.apply_auto_wall_tiling()
        ##return new_state
        return self._replace(
            current_phase=current_phase,
            current_turn=current_turn,
        )

    def player_select_floor_line(self, color: int, place_count: int) -> Self:
        """Return new state after player adds tiles to floor line."""
        assert self.current_phase == Phase.factory_offer
        cursor_contents = self.cursor_contents.copy()
        assert place_count > 0
        assert place_count <= cursor_contents[color]

        box_lid = self.box_lid.copy()
        current_player_data = self.player_data[self.current_turn]

        # Remove from cursor
        cursor_contents[color] -= place_count
        # Add to floor line
        new_player_data, for_box_lid = (
            current_player_data.place_floor_line_tiles(
                color,
                place_count,
            )
        )
        # Add overflow tiles to box lid
        assert all(x > 0 for x in for_box_lid.values()), for_box_lid
        box_lid.update(for_box_lid)

        # If has number one tile, add to floor line
        if cursor_contents[Tile.one]:
            # Add to floor line
            new_player_data, for_box_lid = (
                new_player_data.place_floor_line_tiles(
                    Tile.one,
                    cursor_contents.pop(Tile.one),
                )
            )
            # Add overflow tiles to box lid
            assert all(x > 0 for x in for_box_lid.values()), for_box_lid
            box_lid.update(for_box_lid)

        remove_counter_zeros(cursor_contents)

        # Update player data
        player_data = player_data_deepcopy(self.player_data)
        player_data[self.current_turn] = new_player_data

        return self._replace(
            box_lid=box_lid,
            cursor_contents=cursor_contents,
            player_data=player_data,
        )._factory_offer_maybe_next_turn()

    def player_selects_pattern_line(
        self,
        line_id: int,
        place_count: int,
    ) -> Self:
        """Return new state after player selects line."""
        assert self.current_phase == Phase.factory_offer
        assert not self.is_cursor_empty()
        color = self.get_cursor_holding_color()

        assert self.can_player_select_line(line_id, color, place_count)
        current_player_data = self.player_data[self.current_turn]

        new_player_data = current_player_data.place_pattern_line_tiles(
            line_id,
            color,
            place_count,
        )

        cursor_contents = self.cursor_contents.copy()
        cursor_contents[color] -= place_count

        # Might need to change box lid
        box_lid = self.box_lid

        # If has number one tile, add to floor line
        if cursor_contents[Tile.one]:
            # Will be mutating box lid then
            box_lid = self.box_lid.copy()
            # Add to floor line
            new_player_data, for_box_lid = (
                new_player_data.place_floor_line_tiles(
                    Tile.one,
                    cursor_contents.pop(Tile.one),
                )
            )
            # Add overflow tiles to box lid
            assert all(x > 0 for x in for_box_lid.values()), for_box_lid
            box_lid.update(for_box_lid)

        remove_counter_zeros(cursor_contents)

        player_data = player_data_deepcopy(self.player_data)
        player_data[self.current_turn] = new_player_data

        return self._replace(
            box_lid=box_lid,
            player_data=player_data,
            cursor_contents=cursor_contents,
        )._factory_offer_maybe_next_turn()

    def yield_selectable_tile_destinations_factory_offer(
        self,
    ) -> Generator[SelectableDestinationTiles, None, None]:
        """Yield selectable tile destinations for factory offer phase."""
        assert self.current_phase == Phase.factory_offer
        assert not self.is_cursor_empty()

        current_player_data = self.player_data[self.current_turn]

        color = self.get_cursor_holding_color()
        count = self.cursor_contents[color] + 1

        for (
            line_id,
            placable,
        ) in current_player_data.yield_possible_placement_rows(color):
            for place_count in range(1, min(count, placable + 1)):
                yield SelectableDestinationTiles(
                    destination=SelectableDestination.pattern_line,
                    place_count=place_count,
                    destination_id=line_id,
                )
        # Can always place in floor line, even if full,
        # because of box lid overflow
        for place_count in range(1, count):
            yield SelectableDestinationTiles(
                destination=SelectableDestination.floor_line,
                place_count=place_count,
            )

    def apply_destination_select_action_factory_offer(
        self,
        selection: SelectableDestinationTiles,
    ) -> Self:
        """Return new state after applying destination selection action."""
        assert self.current_phase == Phase.factory_offer
        assert not self.is_cursor_empty()

        if selection.destination == SelectableDestination.floor_line:
            color = self.get_cursor_holding_color()
            return self.player_select_floor_line(
                color,
                selection.place_count,
            )
        if selection.destination == SelectableDestination.pattern_line:
            assert selection.destination_id is not None
            return self.player_selects_pattern_line(
                selection.destination_id,
                selection.place_count,
            )
        raise NotImplementedError(selection.destination)

    def apply_auto_wall_tiling(self) -> Self:
        """Return new state after performing automatic wall tiling."""
        assert self.current_phase == Phase.wall_tiling
        assert not self.variant_play
        box_lid = self.box_lid.copy()
        new_players = player_data_deepcopy(self.player_data)

        is_end = False
        current_turn = self.current_turn
        for player_id, player in self.player_data.items():
            new_player, for_box_lid, has_one = (
                player.perform_auto_wall_tiling()
            )
            new_players[player_id] = new_player
            box_lid.update(for_box_lid)
            if not is_end:
                is_end = new_player.has_horizontal_wall_line()
            if has_one:
                current_turn = player_id

        bag = self.bag.copy()
        factory_displays: dict[int, Counter[int]] = {}

        if is_end:
            for player_id in self.player_data:
                new_players[player_id] = new_players[
                    player_id
                ].perform_end_of_game_scoring()
        else:
            out_of_tiles = False
            for factory_id in self.factory_displays:
                tiles: Counter[int] = Counter()
                if out_of_tiles:
                    factory_displays[factory_id] = tiles
                    continue
                for _ in range(4):
                    if bag.total() > 0:
                        tiles[bag_draw_tile(bag)] += 1
                    else:
                        bag = box_lid
                        box_lid = Counter()
                        if bag.total() <= 0:
                            # "In the rare case that you run out of
                            # tiles again while there are one left in
                            # the lid, start the new round as usual even
                            # though not all Factory displays are
                            # properly filled."
                            out_of_tiles = True
                            break
                factory_displays[factory_id] = tiles

        return self._replace(
            current_phase=Phase.end if is_end else Phase.factory_offer,
            current_turn=current_turn,
            player_data=new_players,
            bag=bag,
            box_lid=box_lid,
            factory_displays=factory_displays,
            table_center=Counter({Tile.one: 1}),
        )

    def get_win_order(self) -> list[tuple[int, int]]:
        """Return player ranking with (id, compare_score) entries."""
        counts: dict[int, int] = {}
        # get_filled_horizontal_line_count can return at most 5
        for player_id, player in self.player_data.items():
            counts[player_id] = (
                player.score * 6 + player.get_filled_horizontal_line_count()
            )
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    def yield_all_factory_offer_destinations(
        self,
    ) -> Generator[tuple[SelectableDestinationTiles, ...]]:
        """Yield all factory offer destinations."""
        if self.is_cursor_empty():
            yield ()
        else:
            for (
                destination
            ) in self.yield_selectable_tile_destinations_factory_offer():
                new = self.apply_destination_select_action_factory_offer(
                    destination,
                )
                did_not_iter = True
                for action in new.yield_all_factory_offer_destinations():
                    did_not_iter = False
                    yield (destination, *action)
                if did_not_iter:
                    yield (destination,)

    def yield_actions(
        self,
    ) -> Generator[
        tuple[SelectableDestinationTiles, ...]
        | tuple[SelectableSourceTiles, tuple[SelectableDestinationTiles, ...]],
        None,
        None,
    ]:
        """Yield all possible actions able to be performed on this state."""
        if self.current_phase == Phase.factory_offer:
            if not self.is_cursor_empty():
                yield from self.yield_all_factory_offer_destinations()
            else:
                for selection in self.yield_selectable_tiles_factory_offer():
                    new = self.apply_source_select_action_factory_offer(
                        selection,
                    )
                    for (
                        action_chain
                    ) in new.yield_all_factory_offer_destinations():
                        yield (selection, action_chain)
        else:
            raise NotImplementedError()

    def preform_action(
        self,
        action: (
            tuple[SelectableDestinationTiles, ...]
            | tuple[
                SelectableSourceTiles,
                tuple[SelectableDestinationTiles, ...],
            ]
        ),
    ) -> Self:
        """Return new state after applying an action."""
        if self.current_phase == Phase.factory_offer:
            if isinstance(action[0], SelectableDestinationTiles):
                new = self
                for destination in action:
                    assert isinstance(destination, SelectableDestinationTiles)
                    new = new.apply_destination_select_action_factory_offer(
                        destination,
                    )
                return new
            selection, destinations = action
            assert isinstance(selection, SelectableSourceTiles)
            new = self.apply_source_select_action_factory_offer(
                selection,
            )
            for destination_ in destinations:
                assert isinstance(destination_, SelectableDestinationTiles)
                new = new.apply_destination_select_action_factory_offer(
                    destination_,
                )
            return new
        raise NotImplementedError()

    def _manual_wall_tiling_maybe_next_turn(self) -> Self:
        # return self
        raise NotImplementedError()

    def get_manual_wall_tiling_locations_for_player(
        self,
        player_id: int,
    ) -> tuple[int, list[int]] | None | Self:
        """Either return player wall tiling location data or new state.

        New state when player cannot wall tile their current row.
        """
        current_player_data = self.player_data[player_id]

        try:
            return current_player_data.get_manual_wall_tile_location()
        except UnplayableTileError as unplayable_exc:
            # kind of hacky, but it works
            y_position = unplayable_exc.y

            new_player_data, for_box_lid = (
                current_player_data.handle_unplayable_wall_tiling(y_position)
            )

            box_lid = self.box_lid.copy()

            # Add overflow tiles to box lid
            assert all(x > 0 for x in for_box_lid.values()), for_box_lid
            box_lid.update(for_box_lid)

            # Update player data
            player_data = player_data_deepcopy(self.player_data)
            player_data[player_id] = new_player_data

            return self._replace(
                box_lid=box_lid,
                player_data=player_data,
            )._manual_wall_tiling_maybe_next_turn()

    def manual_wall_tiling_action(
        self,
        player_id: int,
        line_id: int,
        x_pos: int,
    ) -> Self:
        """Perform manual wall tiling action."""
        current_player_data = self.player_data[player_id]

        new_player_data, for_box_lid = (
            current_player_data.manual_wall_tiling_action(line_id, x_pos)
        )
        box_lid = self.box_lid.copy()

        # Add overflow tiles to box lid
        assert all(x > 0 for x in for_box_lid.values()), for_box_lid
        box_lid.update(for_box_lid)

        # Update player data
        player_data = player_data_deepcopy(self.player_data)
        player_data[player_id] = new_player_data

        new_state = self._replace(
            box_lid=box_lid,
            player_data=player_data,
        )

        result = new_state.get_manual_wall_tiling_locations_for_player(
            player_id,
        )
        if not isinstance(result, self.__class__):
            return new_state._manual_wall_tiling_maybe_next_turn()
        return result._manual_wall_tiling_maybe_next_turn()


def run() -> None:
    """Run program."""
    from market_api import pretty_print_response as pprint

    random.seed(0)
    state = State.new_game(2)
    ticks = 0
    try:
        ##        last_turn = -1
        while state.current_phase == Phase.factory_offer:
            ##            assert last_turn != state.current_turn
            ##            last_turn = state.current_turn
            actions = tuple(state.yield_actions())
            print(f"{len(actions) = }")
            # S311 Standard pseudo-random generators are not suitable
            # for cryptographic purposes
            action = random.choice(actions)  # noqa: S311
            ##            pprint(action)
            state = state.preform_action(action)

            ticks += 1
        print(f"{state.get_win_order() = }")
    except BaseException:
        print(f"{ticks = }")
        ##        print(f'{state = }')
        pprint(state)
        raise
        ##            print(f'{destination = }')
    ##        pprint(state)
    pprint(state)


if __name__ == "__main__":
    run()
