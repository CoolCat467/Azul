#!/usr/bin/env python3
# Client Tile

"""Client Tile"""

# Programmed by CoolCat467

__title__ = "Client Tile"
__author__ = "CoolCat467"
__version__ = "0.0.0"

from client_sprite import ClientSprite

##class


class Tile(ClientSprite):
    def __init__(self, color, *groups):
        super().__init__(*groups)


def run():
    pass


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
    run()
