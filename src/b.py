from PyQt5.QtBluetooth import QBluetoothDeviceDiscoveryAgent, QBluetoothUuid, QBluetoothSocket, QBluetoothAddress, QBluetoothServiceInfo
from PyQt5.QtCore import QIODevice, QTimer
from PyQt5.QtWidgets import QApplication
import sys
import signal

mac_address = "B8:D6:1A:6B:32:7A"
service_uuid = QBluetoothUuid(QBluetoothUuid.SerialPort)
socket = QBluetoothSocket(QBluetoothServiceInfo.RfcommProtocol)

app = QApplication(sys.argv)

def socket_state_changed(state):
    if state == QBluetoothSocket.UnconnectedState:
        print("Socket unconnected")
    elif state == QBluetoothSocket.ConnectingState:
        print("Socket connecting")
    elif state == QBluetoothSocket.ConnectedState:
        print("Socket connected")
socket.stateChanged.connect(socket_state_changed)
socket.connectToService(QBluetoothAddress(mac_address), service_uuid)

def test():
  try:
    print(eval(str(socket.readLine(), "utf-8")))
  except:
    pass

ecg_timer = QTimer()
ecg_timer.timeout.connect(lambda: test())
ecg_timer.start(2)



app.exec_()
