#!/usr/bin/env python3
# Stolen and modified from https://github.com/Dinnerbone/mcstatus
# -*- coding: utf-8 -*-

import socket
from urllib.parse import urlparse
from ipaddress import ip_address
import dns.resolver

def ip_type(address):
    """Returns what version of ip a given address is."""
    try:
        return ip_address(address).version
    except ValueError:
        return None

def parse_address(address):
    """Return a tuple of (address, port) from address string, such as "127.0.0.1:8080" --> ('127.0.0.1', 8080)"""
    tmp = urlparse('//'+address)
    if not tmp.hostname:
        raise ValueError("Invalid address '%s'" % address)
    return (tmp.hostname, tmp.port)

def lookup(address, defaultPort=25565):
    """Look up address, and return MinecraftServer instance after sucessfull lookup."""
    host, port = parse_address(address)
    if port is None:
        port = defaultPort
        try:
##            answers = dns.resolver.resolve("_minecraft._tcp." + host, "SRV")
            answers = dns.resolver.resolve(host)
            if len(answers):
                answer = answers[0]
                host = str(answer.target).rstrip(".")
                port = int(answer.port)
        except Exception:
            pass
    return host, port

# Stolen from WOOF (Web Offer One File), Copyright (C) 2004-2009 Simon Budig,
# avalable at http://www.home.unix-ag.org/simon/woof

# Utility function to guess the IP (as a string) where the server can be
# reached from the outside. Quite nasty problem actually.

def find_ip():
    """Utility function to guess the IP where the server can be found from the network."""
    # we get a UDP-socket for the TEST-networks reserved by IANA.
    # It is highly unlikely, that there is special routing used
    # for these networks, hence the socket later should give us
    # the ip address of the default route.
    # We're doing multiple tests, to guard against the computer being
    # part of a test installation.
    
    candidates = []
    for test_ip in ['192.0.2.0', '198.51.100.0', '203.0.113.0']:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((test_ip, 80))
        ip_addr = sock.getsockname()[0]
        sock.close()
        if ip_addr in candidates:
            return ip_addr
        candidates.append(ip_addr)
    return candidates[0]
