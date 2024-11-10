#!/usr/bin/env python3
# Socket Client Test
# -*- coding: utf-8 -*-

import os, socket

HOST = '127.0.0.1'
PORT = 12345
BUFSIZE = 1040

def run():
    # Initialize the socket
    s = socket.socket()
    # Setup the timeout time to none so it doesn't time out
    s.settimeout(15)

    # Connect the socket to a remote address and return
    # error codes if there is an error
    print('Attempting connection to %s:%i' % (HOST, PORT))
    connErrCode = s.connect_ex((HOST, PORT))

    # If there was an error,
    if connErrCode:
        # Print it and don't continue
        print('Error: '+os.strerror(connErrCode), file=os.sys.stderr)
        input('Press Return to Continue.\n')
        os.abort()
    else:
        # Othewise,
        print('Connection established!\nWaiting for messages...')
        while True:
            # Receive up to buffersize bytes from the socket.
            try:
                rcvdData = s.recv(BUFSIZE).decode()
            except socket.timeout:
                rcvdData = ''
            # Print the recieved transmission
            print('Recieve: ' + '\n'.join(rcvdData.split(';'))[:-1])
            # If the transmission is the word 'bye', exit loop
            if rcvdData.lower() == 'bye':
                break
            elif 'bye' in sum([i.split(' ') for i in rcvdData.split(';')], []):
                break
            elif rcvdData == '':
                print('Server probrally died. Quitting...')
                break
            # Get an input of what to send
            sendData = input('Send: ')
            # Send the data after you encode it
            s.sendall(sendData.encode())
            # If we said 'bye', exit the loop
            if sendData.lower() == 'bye':
                break
            
        # Try to inform the server you are leaving, but don't ensure it happened.
        try:
            s.send(b'bye')
        except BaseException:
            pass
    # Close the socket.
    s.close()

if __name__ == '__main__':
    run()
