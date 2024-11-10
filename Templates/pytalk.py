#!/usr/bin/env python3
# Pytalk; Simple Chat between multiple computers
# -*- coding: utf-8 -*-

# Programmed by CoolCat467

# Utilizes nearly everything in
# https://github.com/Cat-Software-Ink/Socket-Chat

# Uses find_ip function, stolen from WOOF (Web Offer One File),
# Copyright (C) 2004-2009 Simon Budig, avalable at
# http://www.home.unix-ag.org/simon/woof

import os, socket, time
from threading import Thread, Event

__title__ = 'Pytalk'
__author__ = 'CoolCat467'
__version__ = '0.1.0'
__ver_major__ = 0
__ver_minor__ = 1
__ver_patch__ = 0

NONLOCAL = True#False for testing

HOST = '127.0.0.1'
PORT = 30654
BUFSIZE = 1040
TIMEOUT = 120

class serverClient(Thread):
    """Client handling, given the socket, a name to use, and a server we belong to."""
    def __init__(self, socket, addr, name, server):
        Thread.__init__(self, name='serverClient')
        self.socket = socket
        self.addr = addr
        self.name = name
        self.server = server
        
        self.active = False
        self.stopped = False
        self.recvData = b''
        self.start()
    
    def run(self):
        try:
            self.active = True
            while self.active:
                try:
                    self.recvData = self.socket.recv(BUFSIZE)
                except OSError:
                    self.stop()
                else:
                    if not self.recvData or self.recvData == b'':
                        self.stop()
                    else:
                        self.chat(self.recvData.decode('utf-8'))
        finally:
            self.close()
    
    def stop(self):
        """Set self.active to False."""
        self.active = False
        self.server.clientLeft(self.name)
    
    def close(self):
        """Completely close self.socket."""
        try:
            self.socket.shutdown(0)
        except OSError:
            pass
        finally:
            self.socket.close()
            self.stopped = True
            self.server.log('Client %s: Connection Terminated' % self.name)
    
    def chat(self, message):
        """Adds message to self.server.chat."""
        self.server.clientSentMsg(self.name, str(message))
    
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
        while self.server.active:
            # Accept any new connections
            try:
                clientSocket, addr = self.server.socket.accept()
            except OSError as e:
                if not str(e) == '[Errno 22] Invalid argument':
                    self.server.log('AcceptClients: Error: %s' % str(e))
                break
            if not self.server.active:
                break
            
            # Get the new client's IP Address
            ip = addr[0]
            
            # Get the name for this new client
            newCid = str(self.server.nextCid)
            self.server.nextCid += 1
            
            self.server.log('%s (%s) Joined Server.' % (newCid, addr[0]+':'+str(addr[1])))
            # Tell clients about new client
            for cid in self.server.clients:
                self.server.clients[cid].send_all('S %s Joined;' % newCid)
            
            # Add client's address to cidToAddr dictionary
            self.server.cidToAddr[newCid] = ip
            
            # Add client to clients dictionary
            self.server.clients[newCid] = serverClient(clientSocket, addr, newCid, self.server)
            
            # Tell new client about other clients
            self.server.clients[newCid].send_all(
                'S You: %s Others: [%s];' % (newCid, '/'.join([cid for cid in self.server.clients if cid != newCid])))
        self.server.log('AcceptClients: No longer accepting clients.')
    pass

class Server(Thread):
    """Start a new server thread on <host>:<port>."""
    def __init__(self, host, port):
        Thread.__init__(self, name='Server')
        self.host = host
        self.port = port
        
        self.clientLeaveMsg = 'S Connection Terminated'
        
        self.socket = None
        self.ipAddr = None
        self.active = False
        self.stopped = False
        
        self.clients = {}
        self.nextCid = 0
        self.cidToAddr = {}
        
        self.chat = []
        self.logs = []
        self.logging = False
        
        self.start()
    
    def __repr__(self):
        return '<Server Object>'
    
    def log(self, data):
        """Prints data."""
        self.logs.append(data)
##        if self.doPrint:
##            while self.logging:
##                time.sleep(0.01)
##            self.logging = True
##            print('Server: %s' % str(data))
##            self.logging = False
    
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
            # Allow no backlog to exist. All connections should be accepted by AcceptClients thread.
            self.socket.listen(0)
            self.ipAddr = ':'.join([str(i) for i in self.socket.getsockname()])
            self.log('Bound to address successfully.')
    
    def stop(self):
        """Shuts down the server."""
        if not self.stopped:
            self.log('Shutting down server...')
            self.active = False
            try:
                self.socket.shutdown(0)
            except BaseException:
                pass
            self.socket.close()
            for client in [client for client in self.clients.values() if client.is_alive()]:
                client.send_all('S Server shutting down;')
                client.stop()
            time.sleep(0.5)
            if True in [client.is_alive() or client.active for client in self.clients.values()]:
                try:
                    os.wait()
                except ChildProcessError as e:
                    self.log('Error: %s' % e)
            self.stopped = True
            self.log('Server shut down.')
        else:
            self.log('Server already shut down!')
    
    def clientSentMsg(self, clientId, message):
        """Function serverClient threads call to add a message to self.chat."""
        self.chat.append([clientId, message])
    
    def clientLeft(self, clientId):
        """Function serverClient threads call to indicate they are closing."""
        self.clientSentMsg(clientId, self.clientLeaveMsg)
    
    def sendMessageToClient(self, messageWithFromAddr, toCid, log=True):
        """Send a given message with a from address line to a given client."""
        if toCid in self.clients:
            if not messageWithFromAddr.endswith(';'):
                messageWithFromAddr += ';'
            self.clients[toCid].send_all(messageWithFromAddr)
            if log:
                self.log('Send message "%s" to client %s.' % (messageWithFromAddr, toCid))
        else:
            self.log('Cannot send message "%s" to client %s, client does not exist!' % (messageWithFromAddr, toCid))
            raise KeyError('Client %s does not exist!' % toCid)
    
    def forwardMessageToClient(self, fromCid, message, toCid, log=True):
        """Forward a given message from given from client id to given to client id."""
        if toCid in self.clients:
            self.sendMessageToClient(fromCid+' '+message, toCid, False)
            if log:
                self.log('Forwarded client %s\'s message "%s" to client %s.' % (fromCid, message, toCid))
        else:
            self.log('Cannot forward message "%s" to client %s, client does not exist!' % (message, toCid))
            raise KeyError('Client %s does not exist!' % toCid)
    
    def forwardMessageToAllClients(self, fromCid, message, log=True):
        """Forward message <message> from client <fromCid> to all active clients."""
        for client in self.clients:
            if client != fromCid:
                self.forwardMessageToClient(fromCid, message, client, False)
        if log:
            self.log('Forwarded client %s\'s message "%s" to all clients.' % (fromCid, message))
    
    def processCommand(self, fromCid, command):
        """Process commands sent to server."""
        if ' ' in command:
            cdata = command.split(' ')
        elif command == '' or len(command) == 0 or command is None or not fromCid in self.clients:
            return
        else:
            cdata = [command]
        validClients = (cid for cid in self.clients if cid != fromCid)
        cmd = cdata[0]
        args = cdata[1:]
        self.log('Processing command "%s" from client %s.' % (command, fromCid))
        
        if cmd == 'kick':
            if len(args) > 0:
                if not 'A' in args:
                    validClients = [cid for cid in validClients if cid in args and cid != '0']
                else:
                    validClients = [cid for cid in validClients if cid != '0']
                for cid in validClients:
                    self.sendMessageToClient('S Client %s kicked you from the server;' % fromCid, cid, False)
                    self.clients[cid].close()
                    self.log('Kicked client %s.' % cid)
                self.sendMessageToClient('S Successfully kicked %i client(s);' % len(validClients), fromCid, False)
            elif fromCid != '0':
                self.sendMessageToClient('S Kicking you from the server;', fromCid, False)
                self.clients[fromCid].close()
                self.log('Kicked client %s at their request.' % fromCid)
            else:
                self.sendMessageToClient('S You, being the host of the server, are not allowed to kick yourself. Press CTRL+C to shut down.;', '0', False)
                self.log('Client 0 requested to be kicked. Denying request, as client zero is always the server host.')
        elif cmd == 'list':
            self.sendMessageToClient('S You: %s Others: [%s]' % (fromCid, '/'.join(iter(validClients))), fromCid, False)
            self.log('Told client %s about connected users.' % fromCid)
        elif cmd == 'help':
            self.sendMessageToClient(''.join('''S Command Help: Commands:;
"kick <cid>": Kicks a client from the server. Blank kicks yourself from the server;
"list": Lists connected clients;
"help": Server sends you this message;'''.splitlines()), fromCid, False)
            self.log('Client %s requested help message.' % fromCid)
        else:
            # If nothing has already proccessed a command,
            # then the command is invalid
            self.log('Client %s sent an invalid command.' % fromCid)
            self.sendMessageToClient('S Invalid command. Use "help" to list valid commands.', fromCid, False)
    
    def processChat(self):
        """Read chat messages and act apon them."""
        clientsToDelete = []
        if not self.chat:
            time.sleep(0.1)
            return
        for cidx in reversed(range(len(self.chat))):
            fromCid, clientMsgs = self.chat[cidx]
            # Messages are split by semicolons.
            for clientMsg in clientMsgs.split(';'):
                # If message is blank (end simicolons or something), ignore message.
                if clientMsg == '':
                    continue
                # If the client that sent the message is still active
                if fromCid in self.clients:
                    # If the client sent the client leave message, delete that client
                    if clientMsg == self.clientLeaveMsg:
                        if not fromCid in clientsToDelete:
                            clientsToDelete.append(fromCid)
                    # Otherwise, see if the client is sending a message to another client
                    elif ' ' in clientMsg:
                        self.log('Recieved message "%s" from client id %s.' % (clientMsg, fromCid))
                        splitMsg = clientMsg.split(' ')
                        toCid = str(splitMsg[0])
                        # Get the message they sent
                        baseMsg = ' '.join(splitMsg[1:])
                        del splitMsg
                        # Check if to client id is valid
                        if toCid in self.clients:
                            if not toCid in clientsToDelete:
                                self.forwardMessageToClient(fromCid, baseMsg, toCid)
                        elif toCid == 'S':
                            self.processCommand(fromCid, baseMsg)
                        elif toCid == 'A':
                            self.forwardMessageToAllClients(fromCid, baseMsg)
                        else:
                            self.forwardMessageToAllClients(fromCid, clientMsg)
##                            self.log('Client %s tried to send a message to an invalid client id "%s".' % (fromCid, toCid))
##                            self.sendMessageToClient('S Could not send message to %s, invalid client id;' % toCid, fromCid)
                    else:
                        # If no send address specified, send to all.
                        self.forwardMessageToAllClients(fromCid, clientMsg)
##                        self.log('Client %s sent an invalid message; Telling them.' % fromCid)
##                        if fromCid in self.clients.keys():
##                            self.clients[fromCid].send_all('S Invalid message;')
            del self.chat[cidx]
        
        if clientsToDelete:
            for client in clientsToDelete:
                self.clients[client].close()
                del self.clients[client]
            for cid in clientsToDelete:
                self.forwardMessageToAllClients('S', '%s Left;' % cid, False)
            self.log('All users informed of the leaving of user(s) %s.' % ' ,'.join(clientsToDelete))
    
    def run(self):
        """Begins accepting clients and proccessing chat data."""
        self.startSocket()
        try:
            if self.active:
                self.log('Server up and running on %s!' % self.ipAddr)
                AcceptClients(self)
                while self.active:
                    self.processChat()
        except BaseException as e:
            self.log('Error: %s' % str(e))
        finally:
            self.stop()
    pass

class Client(Thread):
    """Thread that, while active, continuously reads messages into self.chat."""
    def __init__(self, host, port, timeout=15, doPrint=False):
        Thread.__init__(self, name='Client')
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.doPrint = bool(doPrint)
        
        self.socket = None
        self.active = False
        self.stopped = False
        
        self.chat = []
        
        self.start()
    
    def log(self, message):
        """Logs a message if self.doPrint is True."""
        self.chat.append(message)
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
        try:
            connErrCode = self.socket.connect_ex((self.host, self.port))
        except socket.gaierror:
            connErrCode = -3
        if connErrCode:
            self.log('Error: '+os.strerror(connErrCode))
        else:
            self.active = True
            self.log('Connection established!')
    
    def stop(self):
        """Close self.socket."""
        if not self.stopped:
            self.log('Shutting down...')
            try:
                self.socket.shutdown(0)
            except OSError:
                pass
            finally:
                self.socket.close()
                self.log('Socket closed.')
                self.stopped = True
        else:
            self.log('Already shut down!')
    
    def send(self, message):
        """Encodes message and sends all on self.socket."""
        if self.stopped:
            raise OSError('Socket closed!')
        if not message.endswith(';'):
            message += ';'
        try:
            self.socket.sendall(message.encode('utf-8'))
        except OSError:
            self.active = False
    
    def recieve(self):
        """Returns decoded messages from self.socket."""
        try:
            rcvdData = self.socket.recv(BUFSIZE)
        except socket.timeout:
            self.log('Connection Timed Out.')
            return ''
        except BaseException:
            return ''
        return rcvdData.decode('utf-8')
    
    def run(self):
        """Recieved data and stores individual messages in self.chat."""
        try:
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
                    self.log(msg)
        finally:
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
    args = os.sys.argv[1:]
    
    definedServer = False
    doServer = True
    showServerMessages = False
    
    kwargs = {k:v for k, v in [a.split('=') for a in args if '=' in a]}
    
    if '-h' in args or '-help' in args or '--help' in args:
        print("""Usage:
\t-h : Show help message and exit.
\tmode=[0, 1] : Set mode to Server (0) or Client (1) mode. Can be overwritten by other arguments.
\tlocal=[True, False] : Force IP to use to be local or use public IP if not given.
\tip=<IP Address> : Set IP to given value and set mode to Client mode.
\tport=<Port> : Set Port to connect/host on to given port.
\ttimeout=<Timeout> : Set Timeout for client and set mode to Client mode.
\tshowServerMessages : Show server messages. Mostly for debug purposes. Only has effect when in server mode.\n
If no arguments with equals (keyword arguments) are given, first argument if given is IP to connect to and mode is automatically set to Client mode.
If two arguments are given, first argument is IP address to connect to and seccond argument is custom Port to connect to at given IP address and mode is automatically set to Client mode.
If no arguments are given at all, mode is automatically set to Server mode.""")
        os.sys.exit(0)
    
    if 'mode' in kwargs:
        if kwargs['mode'] in ('0', '1'):
            definedServer = True
            doServer = {'0':True, '1':False}[kwargs['mode']]
    
    if 'local' in kwargs:
        if kwargs['local'] in ('True', 'False'):
            NONLOCAL = {'True':False, 'False':True}[kwargs['local']]
    
    if 'ip' in kwargs and not definedServer:
        HOST = kwargs['ip']
        definedServer = True
        doServer = False
    elif NONLOCAL:
        HOST = find_ip()
    
    if 'port' in kwargs:
        PORT = int(kwargs['port'])
    if 'timeout' in kwargs and not definedServer:
        TIMEOUT = int(kwargs['timeout'])
        definedServer = True
        doServer = False
    
    if not kwargs:
        if args:
            if 'showServerMessages' in args:
                showServerMessages = True
            else:
                l = len(args)
                if l == 1:
                    HOST = args[0]
                    definedServer = True
                    doServer = False
                elif l == 2:
                    HOST, PORT = args[:2]
                    definedServer = True
                    doServer = False
        else:
            definedServer = True
            doServer = True
    del definedServer, args, kwargs
    
    def seeKill(lst):
        for i in reversed(range(len(lst))):
            yield lst[-(i+1)]
            del lst[-(i+1)]
    
    if doServer:
        print('\nWe will be attempting to host the server.')
        server = Server(HOST, PORT)
        while not server.active and not server.stopped:
            time.sleep(0.1)
        if server.stopped:
            print('\n')
            msgs = list(server.logs)
            if not 'Error: [Errno 98] Address already in use' in msgs:
                for message in seeKill(server.logs):
                    print('Server: '+message)
                print('\nError: Server stopped!')
                print('Would you like to initalize a client anyways?')
                if input('(y/n) : ').lower() in ('y', 'yes'):
                    doServer = False
                else:
                    os.sys.exit(1)
            else:
                if showServerMessages:
                    for message in seeKill(server.logs):
                        print('Server: '+message)
                else:
                    list(seeKill(server.logs))
                print('Error: Cannot host server: Address already in use.')
                print('Attempting connection.')
                doServer = False
                del msgs
    
    client = Client(HOST, PORT, TIMEOUT)
    while not client.active and not client.stopped:
        time.sleep(0.1)
    if client.stopped:
        print('\nError: Client stopped!')
    seen = []
    
    print('\nPress CTRL+C to quit.\nPress Return without typing anything in to show new messages.\n')
    
    if doServer:
        print('Note: Connecting to the server we are hosting.')
    
    try:
        while client.active:
            if doServer:
                if not server.active:
                    client.active = False
                    print('Server just died.')
                elif showServerMessages:
                    for message in seeKill(server.logs):
                        print('Server: '+message)
                else:
                    list(seeKill(server.logs))
            for message in seeKill(client.chat):
                print('Client: '+message)
            try:
                tosend = input('Send  : ')
                if tosend != '':
                    client.send(tosend+';')
            except BaseException:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        client.active = False
        print('\nClosing program...\n')
    finally:
        client.active = False
        if doServer:
            server.active = False
    
    if doServer:
        while not server.stopped:
            time.sleep(0.1)
        for message in seeKill(server.logs):
            print('Server: '+message)
    
    try:
        while not client.stopped:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('Stopping Client...')
        client.stop()
        while not client.stopped:
            time.sleep(0.1)
    for message in seeKill(client.chat):
        print('Client: '+message)
