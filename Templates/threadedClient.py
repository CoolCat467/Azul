#!/usr/bin/env python3
# Threaded Client allowing sending and recieveing of messages at the same time
# -*- coding: utf-8 -*-

# Programmed by CoolCat467

import os, socket, time
from threading import Thread, Event

__title__ = 'Threaded Client'
__author__ = 'CoolCat467'
__version__ = '0.0.0'
__ver_major__ = 0
__ver_minor__ = 0
__ver_patch__ = 0

HOST = '127.0.0.1'
PORT = 12345
BUFSIZE = 1040

class Client(Thread):
    """Thread that, while active, continuously reads messages into self.chat."""
    def __init__(self, host, port, timeout=15, doPrint=False):
        Thread.__init__(self, name='Client Thread')
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.doPrint = bool(doPrint)
        self.socket = None
        self.active = False
        self.chat = []
        self.start()
    
    def log(self, message):
        """Logs a message if self.doPrint is True."""
        if self.doPrint:
            print('Client: %s' % str(message))
    
    def startSocket(self):
        """Initialize the socket and connect to server with given info."""
        # Initialize the socket
        self.socket = socket.socket()
        # Set timeout to given timeout
        self.socket.settimeout(self.timeout)
        # Connect the socket to a remote address and return
        # error codes if there is an error
        self.log('Attempting Connection to %s:%i...' % (self.host, self.port))
        connErrCode = self.socket.connect_ex((self.host, self.port))
        if connErrCode:
            self.doPrint = True
            self.log('Error: '+os.strerror(connErrCode))
        else:
            self.active = True
            self.log('Connection established!')
    
    def stop(self):
        """Close self.socket."""
        self.log('Shutting down...')
        try:
            self.socket.shutdown(0)
        except OSError:
            pass
        self.socket.close()
        self.log('Socket closed.')
    
    def send(self, message):
        """Encodes message and sends all on self.socket."""
        try:
            self.socket.sendall(message.encode('utf-8'))
        except OSError:
            self.active = False
    
    def recieve(self):
        """Returns decoded messages from self.socket."""
        try:
            rcvdData = self.socket.recv(BUFSIZE)
        except socket.timeout:
            rcvdData = b''
        except OSError:
            rcvdData = b''
        return rcvdData.decode('utf-8')
    
    def run(self):
        """Recieved data and stores individual messages in self.chat."""
        self.startSocket()
        while self.active:
            data = self.recieve()
            if data == '':
                self.active = False
                self.log('Connection Terminated. Shutting down...')
                continue
            if not data.endswith(';'):
                data += ';'
            for msg in data.split(';')[:-1]:
                self.log('Recieved message "%s".' % msg)
                self.chat.append(msg)
        self.stop()
    pass

class ClientWithInput(Client):
    """Like a regular client, but always prints and also has an input."""
    def run(self):
        self.doPrint = True
        self.startSocket()
        while self.active:
            data = self.recieve()
            if data == '':
                self.active = False
                self.log('Connection Terminated. Shutting down...')
                continue
            if not data.endswith(';'):
                data += ';'
            for msg in data.split(';')[:-1]:
                self.log('Recieved message "%s".' % msg)
                self.chat.append(msg)
            self.send(input('Send: '))
        self.stop()

if __name__ == '__main__':
    client = ClientWithInput(HOST, PORT, 30)
