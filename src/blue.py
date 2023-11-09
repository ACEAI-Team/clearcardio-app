from PyQt5.QtBluetooth import QBluetoothDeviceDiscoveryAgent, QBluetoothUuid, QBluetoothSocket, QBluetoothAddress, QBluetoothServiceInfo
from PyQt5.QtCore import QIODevice
from PyQt5.QtWidgets import QApplication
import sys

mac_address = "B8:D6:1A:6B:32:7A"  # Replace with your device's MAC address

app = QApplication(sys.argv)

def device_discovered(device):
    if device.address().toString() == mac_address:
        print(f"Found device: {device.name()} ({device.address().toString()})")
        socket = QBluetoothSocket(QBluetoothServiceInfo.RfcommProtocol)
        service_uuid = QBluetoothUuid(QBluetoothUuid.SerialPort)
        socket.connectToService(QBluetoothAddress(mac_address),1)
        if socket.state() == QBluetoothSocket.ConnectedState:
            print("Connected to Bluetooth device!")
            socket.write(QByteArray(b"A"))
            data = socket.readAll()
            print(f"Received data: {data}")
        else:
            print("Failed to connect to Bluetooth device.")
        socket.close()

agent = QBluetoothDeviceDiscoveryAgent()
agent.deviceDiscovered.connect(device_discovered)
agent.start()

sys.exit(app.exec_())

