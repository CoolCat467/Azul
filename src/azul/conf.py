#!/usr/bin/env python3
# Config module

"""Config module."""

# Programmed by CoolCat467

__title__ = "Conf"
__author__ = "CoolCat467"
__version__ = "0.0.0"


from configparser import ConfigParser


def load_config(config_file: str) -> dict[str, str]:
    """Return a config object from config_file."""
    config = ConfigParser()
    config.read((config_file,))

    data = {}
    for section, values in dict(config.items()).items():
        data[section] = dict(values)

    ##    config.clear()
    ##    config.update(data)
    ##
    ##    with open(config_file, mode='w', encoding='utf-8') as conf_file:
    ##        config.write(conf_file)

    return data


if __name__ == "__main__":
    print(f"{__title__}\nProgrammed by {__author__}.")
