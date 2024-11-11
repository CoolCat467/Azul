"""Room."""

# Programmed by CoolCat467

__title__ = "Room"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from gears import AsyncState, StateTimer


class Unloaded(AsyncState):
    """Unloaded state."""

    __slots__ = ()

    def __init__(self):
        """Unloaded state init."""
        super().__init__("unloaded")

    async def check_conditions(self) -> None:
        """Print machine size."""
        print(self.machine.size)


class Lobby(AsyncState):
    """Lobby state. Has host."""

    __slots__ = ()

    def __init__(self):
        """Loby state init."""
        super().__init__("lobby")
        self.machine.host = None

    async def check_conditions(self) -> None:
        """Print machine size."""
        print(self.machine.size)


class BaseRoom(StateTimer):
    """Base room."""

    __slots__ = ("size", "clients", "host")
    gear_type = "room"
    min_delay = 0

    def __init__(self, server, name: str):
        """Room init."""
        super().__init__(server, name, 0)

        self.size = 4

        self.clients = {}
        self.host = None

        self.add_state(Lobby())

    async def initialize_state(self) -> None:
        """Set state to lobby."""
        await self.set_state("lobby")

    def add_client(self, client) -> None:
        """Add client."""
        self.clients[client.id] = client


class Room(BaseRoom):
    """Room."""

    __slots__ = ()

    def __init__(self, server):
        """Room init."""
        super().__init__(server, "room")


class Liminal(BaseRoom):
    """Liminal space. A room between rooms."""

    __slots__ = ()
    gear_type = "liminal_room"

    def __init__(self, server):
        """Liminal room init."""
        super().__init__(server, "limital")


def run():
    """Run."""


if __name__ == "__main__":
    print(f"{__title__} v{__version__}\nProgrammed by {__author__}.")
    run()
