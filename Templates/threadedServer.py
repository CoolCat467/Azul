#!/usr/bin/env python3
# Threaded server allowing multiple connections
# -*- coding: utf-8 -*-

# Programmed by CoolCat467

import os, socket, time
import random #For random password generation
from threading import Thread, Event

__title__ = 'Threaded Server'
__author__ = 'CoolCat467'
__version__ = '0.1.0'
__ver_major__ = 0
__ver_minor__ = 1
__ver_patch__ = 0

# Set up port information
host = '127.0.0.1'#localhost
port = 12345
BUFSIZE = 1040
MAXCONNS = 2

NONLOCAL = True

class Client(Thread):
    """Client handling, given the socket, a name to use, and a server we belong to."""
    def __init__(self, socket, addr, name, server):
        Thread.__init__(self, name='Client')
        self.socket = socket
        self.addr = addr
        self.name = name
        self.server = server

        self.active = False
        self.recvData = None
        self.start()

    def run(self):
        self.active = True
        while self.active and not self.server.closeEvent.isSet():
            try:
                self.recvData = self.socket.recv(BUFSIZE)
            except OSError as e:
                self.active = False
            else:
                if not self.addr[0] in self.server.bannedIps:
                    if not self.recvData or self.recvData == b'' or self.server.closeEvent.isSet():
                        self.active = False
                    else:
                        self.chat(self.recvData.decode('utf-8'))
                else:
                    self.active = False
        self.server.chat.append([self.name, self.server.clientLeaveMsg])
        self.close()
        self.server.log('Client %s: Connection Terminated' % self.name)

    def close(self):
        """Completely close self.socket."""
        try:
            self.socket.shutdown(0)
        except OSError:
            pass
        self.socket.close()

    def chat(self, message):
        """Adds message to self.server.chat."""
        self.server.chat.append([self.name, message])

    def send_all(self, message):
        """Encode message in utf-8 format and sendall to self.socket"""
        if self.active:
            self.socket.sendall(message.encode('utf-8'))
    pass

class AcceptClients(Thread):
    """Thread to accept connections to <server> and create new Client threads for it."""
    def __init__(self, server):
        Thread.__init__(self, name='Server Client Acceptor')
        self.server = server
        self.start()

    def run(self):
        """While the server is active, if we get a connection that's IP is not it the server's bannedIps list, add the connection with a new Client Thread."""
        while server.active and not self.server.closeEvent.isSet():
            if len(self.server.clients) < self.server.maxClients:
                # Accept any new connections
                try:
                    clientSocket, addr = self.server.socket.accept()
                except OSError as e:
                    if not str(e) == '[Errno 22] Invalid argument':
                        self.server.log('AcceptClients: Error: %s' % str(e))
                    break
                if not server.active:
                    break
                # Get the new client's IP Address
                ip = addr[0]#+':'+str(addr[1])
                # If the IP is one of the server's banned IPs,
                if ip in self.server.bannedIps:
                    # Tell that client they're banned and shutdown the connection
                    clientSocket.sendall(b'S You are banned from this server.')
                    clientSocket.shutdown(0)
                    clientSocket.close()
                    # Log this event
                    self.server.log('Banned IP Address "%s" attempted to join server.' % ip)
                    continue

                # Get the name for this new client
                newCid = str(self.server.nextCid)
                self.server.nextCid += 1

                self.server.log('%s (%s) Joined Server.' % (addr, newCid))
                # Tell clients about new client
                for cid in self.server.clients.keys():
                    self.server.clients[cid].send_all('S %s Joined;' % newCid)

                # Add client's address to cidToAddr dictionary
                self.server.cidToAddr[newCid] = ip

                # Add client to clients dictionary
                self.server.clients[newCid] = Client(clientSocket, addr, newCid, self.server)

                # Tell new client about other clients
                self.server.clients[newCid].send_all(
                    'S You: %s Others: [%s];' % (newCid, '/'.join(list(self.server.clients.keys()))))
            else:
                time.sleep(1)
        self.server.log('AcceptClients: No longer accepting clients.')
    pass

class Server(Thread):
    """Start a new server thread on <host>:<port>, with <maxClients> maximum connected clients."""
    def __init__(self, host, port, maxClients, doPrint=False, passwd='rand9'):
        Thread.__init__(self, name='Server')
        self.host = host
        self.port = port
        self.maxClients = int(maxClients)
        self.doPrint = bool(doPrint)
        passwd = str(passwd)
        if len(passwd) >= 4 and passwd[:4] == 'rand':
            if len(passwd) >= 5 and passwd[4:].isdigit():
                cnt = int(passwd[4:])
            else:
                cnt = random.randint(9, 12)
            choose = [chr(i) for i in range(65, 122)]+[str(i) for i in range(10)]
            self.password = ''.join([random.choice(choose) for i in range(cnt)])
        elif not ' ' in passwd and not ';' in passwd:
            self.password = str(passwd)
        else:
            self.password = str(''.join(''.join(passwd.split(' ')).split(';')))

        self.clientLeaveMsg = 'S Connection Terminated'

        self.socket = None
        self.ipAddr = None
        self.active = False
        self.stopped = False

        self.clients = {}
        self.trustedClients = []
        self.bannedIps = []
        self.closeEvent = Event()
        self.nextCid = 0
        self.cidToAddr = {}

        self.chat = []

        self.start()

    def __repr__(self):
        return '<Server Object>'

    def log(self, data):
        """Prints data."""
        if self.doPrint:
            print('Server: %s' % str(data))

    def startSocket(self):
        """Initializes a new socket for the server to work on."""
        self.log('Binding Socket to %s:%i...' % (self.host, self.port))
        self.socket = socket.socket()
        try:
            self.socket.bind((self.host, self.port))
        except OSError as e:
            self.log('Error: %s' % str(e))
        else:
            self.active = True
            # Allow no backlog to exist. All connections should be accepted
            # until we reach max connections, then refuse all.
            self.socket.listen(0)
            self.ipAddr = ':'.join([str(i) for i in self.socket.getsockname()])
            self.log('Bound to address successfully.')

    def stop(self):
        """Shuts down the server."""
        if not self.stopped:
            self.log('Shutting down server...')
            self.active = False
            self.closeEvent.set()
            try:
                self.socket.shutdown(0)
            except OSError:
                pass
            self.socket.close()
            for client in [client for client in self.clients.values() if client.is_alive()]:
                client.send_all('S Server shutting down;')
                client.close()
            time.sleep(0.5)
            if True in [client.is_alive() or client.active for client in self.clients.values()]:
                try:
                    os.wait()
                except ChildProcessError as e:
                    self.log('Error: %s' % e)
            self.stopped = True
        else:
            self.log('Server already shut down!')

    def processCommand(self, fromCid, command):
        """Process commands sent to server."""
        if ' ' in command:
            cdata = command.split(' ')
        elif len(command) == 0 or command is None or not fromCid in self.clients.keys():
            return
        else:
            cdata = [command]
        self.log('Processing command "%s" from client "%s".' % (command, fromCid))
        trusted = fromCid in self.trustedClients
        if cdata[0] == 'shutdown' and trusted:
            self.active = False
            self.log('Trusted Client "%s" issued the shutdown command!' % fromCid)
        elif cdata[0] == 'auth':
            if not trusted:
                if len(cdata) == 2:
                    if cdata[1] == self.password:
                        self.log('Trusting client "%s"' % fromCid)
                        self.trustedClients.append(fromCid)
                        self.clients[fromCid].send_all('S Authentication successfull;')
                    else:
                        self.bannedIps.append(self.cidToAddr[fromCid])
                        self.clients[fromCid].send_all('S You are banned from this server for attempted hacking;')
                        self.clients[fromCid].close()
                        self.log('Banned IP Address "%s" for attempting to authenticate with invalid password.' % self.cidToAddr[fromCid])
                else:
                    self.clients[fromCid].send_all('S Cannot Authenticate without password;')
                    self.log('Client "%s" failed to supply password to authenticate.' % fromCid)
            else:
                self.clients[fromCid].send_all('S You are already authenticated;')
                self.log('Client "%s" is already authenticated.' % fromCid)
        elif cdata[0] == 'ban':
            if trusted:
                if len(cdata) > 1:
                    validClients = [cid for cid in cdata[1:] if cid in self.clients.keys() and cid != fromCid]
                    for cid in validClients:
                        self.bannedIps.append(self.cidToAddr[cid])
                        self.clients[cid].send_all('S You are banned from this server by %s;' % fromCid)
                        self.clients[cid].close()
                        self.log('Banned IP Address "%s".' % self.cidToAddr[cid])
                    self.clients[fromCid].send_all('S Successfully banned %i client(s);' % len(validClients))
                else:
                    self.clients[fromCid].send_all('S Banned 0 clients;')
            else:
                self.clients[fromCid].send_all('S Access Denied;')
        elif cdata[0] == 'kick':
            if trusted and len(cdata) > 1:
                validClients = [cid for cid in cdata[1:] if cid in self.clients.keys() and cid != fromCid]
                for cid in validClients:
                    self.clients[cid].send_all('S %s Kicked you from the server;' % fromCid)
                    self.clients[cid].close()
                    self.log('Kicked client "%s".' % cid)
                self.clients[fromCid].send_all('S Successfully kicked %i client(s);' % len(validClients))
            else:
                self.clients[fromCid].send_all('S Kicking you from the server;')
                self.clients[fromCid].close()
                self.log('Kicked client id "%s" at their request.' % fromCid)
        elif cdata[0] == 'list':
            self.clients[fromCid].send_all('S You: %s Others: [%s] Admins: [%s];' % (fromCid, '/'.join(self.clients.keys()), '/'.join(self.trustedClients)))
            self.log('Told "%s" was about active users.' % fromCid)
        elif cdata[0] == 'help':
            if trusted:
                self.clients[fromCid].send_all(''.join('''S Command Help: Commands:;
"shutdown": Shuts down server;
"ban <client>": Ban <client>'s ip address from server;
"kick <target>": Kicks target from server. Blank sets target to self;
"list": Lists connected clients and admins;
"help": Server sends you this message;'''.splitlines()))
            else:
                self.clients[fromCid].send_all(''.join('''S Command Help: Commands:;
"auth <password>": Authenticates user. Invalid results in ban;
"kick": Kicks yourself from the server;
"list": Lists connected clients and admins;
"help": Server sends you this message;'''.splitlines()))
            self.log('Client "%s" requested help message.' % fromCid)
        else:
            # If nothing has already proccessed a command,
            # then the command is invalid
            self.log('Client "%s" sent an invalid command.' % fromCid)
            self.clients[fromCid].send_all('S Invalid command. Use "help" to list commands.')

    def processChat(self):
        """Read chat messages and act apon them."""
        deletedClients = []
        while self.active:
            clientsToDelete = []
            if not self.chat:
                time.sleep(0.1)
                continue
            for cidx in range(len(self.chat)-1, -1, -1):
                fromCid, clientMsgs = self.chat[cidx]

                # Messages are split by semicolons.
                for clientMsg in clientMsgs.split(';'):
                    self.log('Recieved message "%s" from client id "%s"' % (clientMsg, fromCid))
                    # If the client sent the client leave message, delete that client
                    if clientMsg == self.clientLeaveMsg:
                        if not fromCid in clientsToDelete:
                            if not fromCid in deletedClients:
                                clientsToDelete.append(fromCid)
                    # Otherwise, see if the client is sending a message to another client
                    elif ' ' in clientMsg:
                        toCid = clientMsg.split(' ')[0]
                        # Check if to client id is valid
                        if toCid in self.clients.keys() and not toCid in clientsToDelete:
                            # If to client id is valid, get the message they sent
                            origMsg = ' '.join(clientMsg.split(' ')[1:])
                            sendMsg = fromCid+' '+origMsg
                            if not sendMsg.endswith(';'):
                                sendMsg = str(sendMsg)+';'
                            self.clients[toCid].send_all(sendMsg)
                            self.log('Sent Client "%s" Message "%s";' % (toCid, origMsg))
                        elif toCid == 'S':
                            self.processCommand(fromCid, ' '.join(clientMsg.split(' ')[1:]))
                        elif not fromCid in deletedClients:
                            self.log('Client "%s" tried to send a message to an invalid client id "%s".' % (fromCid, toCid))
                            self.clients[fromCid].send_all('S Could not send message to %s, invalid client id;' % toCid)
                    else:
                        self.log('Client "%s" sent an invalid message.' % fromCid)
                        if fromCid in self.clients.keys():
                            self.clients[fromCid].send_all('S Invalid message;')
                del self.chat[cidx]
            for cid in clientsToDelete:
                if cid in self.trustedClients:
                    del self.trustedClients[self.trustedClients.index(cid)]
                self.clients[fromCid].close()
                deletedClients.append(cid)
                del self.clients[cid]
                for client in iter(self.clients.values()):
                    client.send_all('S %s Left;' % cid)

    def run(self):
        """Begins accepting clients and proccessing chat data."""
        self.startSocket()
        if self.active:
            self.log('Server up and running on %s!' % self.ipAddr)
            self.log('Server Password: %s' % self.password)
            self.log('Accepting up to %i clients at once.' % self.maxClients)
            AcceptClients(self)
            try:
                self.processChat()
            except BaseException as e:
                self.log('Error: %s' % str(e))
        self.stop()
    pass

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
    for test_ip in ["192.0.2.0", "198.51.100.0", "203.0.113.0"]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((test_ip, 80))
        ip_addr = sock.getsockname()[0]
        sock.close()
        if ip_addr in candidates:
            return ip_addr
        candidates.append(ip_addr)

    return candidates[0]

if __name__ == '__main__':
    print('%s Version %s Programmed by %s' % (__title__, __version__, __author__))
    if NONLOCAL:
        host = find_ip()
    server = Server(host, port, MAXCONNS, True, passwd='rand')
