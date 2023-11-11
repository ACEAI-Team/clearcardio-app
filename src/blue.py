from PyQt5.QtBluetooth import QBluetoothDeviceDiscoveryAgent, QBluetoothUuid, QBluetoothSocket, QBluetoothAddress, QBluetoothServiceInfo
from PyQt5.QtCore import QIODevice
from PyQt5.QtWidgets import QApplication
import sys
import signal

mac_address = "B8:D6:1A:6B:32:7A"
service_uuid = QBluetoothUuid(QBluetoothUuid.SerialPort)
socket = QBluetoothSocket(QBluetoothServiceInfo.RfcommProtocol)

app = QApplication(sys.argv)

'''
def sigint_handler(*args):
    print("Program interrupted, disconnecting from the device...")
    socket.close()
signal.signal(signal.SIGINT, sigint_handler)

'''

def connected():
    print("Connected to the Bluetooth device.")
    while 1:
      print(socket.state(), socket.readLine(), end='\r')
socket.connected.connect(connected)
def socket_state_changed(state):
    # handle different socket states
    if state == QBluetoothSocket.UnconnectedState:
        print("Socket unconnected")
    elif state == QBluetoothSocket.ConnectingState:
        print("Socket connecting")
    elif state == QBluetoothSocket.ConnectedState:
        print("Socket connected")
socket.stateChanged.connect(socket_state_changed)
def receivedBluetoothMessage(self):
    while socket.canReadLine():
        line = socket.readLine()
        print(str(line, "utf-8"))
socket.readyRead.connect(receivedBluetoothMessage)

def device_discovered(device):
    if device.address().toString() == mac_address:
        print(f"Found device: {device.name()} ({device.address().toString()})")
        socket.connectToService(QBluetoothAddress(mac_address), service_uuid)
agent = QBluetoothDeviceDiscoveryAgent()
agent.deviceDiscovered.connect(device_discovered)
agent.start()


'''
while 1:
  print(socket.state(), socket.canReadLine(), end = '\r')
'''

'''
print('Attempting Connection...')
while socket.state() != QBluetoothSocket.ConnectingState:
  print(socket.state(), end='\r')
print('Connecting...')
while socket.state() != QBluetoothSocket.ConnectedState:
  print(socket.state(), end='\r')
print("Connected to Bluetooth device!")
print('Waiting...', end='')
while not socket.canReadLine():
  print(socket.canReadLine(), end='\r')
while socket.canReadLine():
  line = str(socket.readLine(), 'utf-8').strip()
  print(f"Received: {line}")
socket.close()
print("Disconnected")
'''

app.exec_()
