#!/usr/bin/env python3
# TITLE DISCRIPTION

"""Docstring"""

# Programmed by CoolCat467

__title__ = "TITLE"
__author__ = "CoolCat467"
__version__ = "0.0.0"

##import zlib


class Packet:
    packet_id = None

    def __init__(self, length, data):
        self.length = length
        self.data = bytearray(data)


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
    run()
