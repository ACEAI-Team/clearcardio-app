import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsTextItem, QGraphicsProxyWidget, QSizePolicy
from PyQt5.QtGui import QPainter, QColor, QPixmap, QPen
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice, QLineSeries, QValueAxis
from PyQt5.QtCore import Qt, QIODevice, QTimer, QThread
from PyQt5.QtBluetooth import QBluetoothDeviceDiscoveryAgent, QBluetoothUuid, QBluetoothSocket, QBluetoothAddress, QBluetoothServiceInfo
import numpy as np
import torch
import model


names = ['Non-ecotopic', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
chances = [50, 20, 15, 10, 5]
colors = [QColor(100, 255, 100), QColor(255, 255, 200), QColor(255, 200, 150), QColor(255, 130, 130), QColor(200, 80, 255)]
ecg_data = np.zeros(300)
ecg_data = np.array([0.15765766,0.14564565,0.16516517,0.04804805,0.16816817,0.14564565,0.17417417,0.17117117,0.15765766,0.14864865,0.16066066,0.04804805,0.17417417,0.14564565,0.16516517,1.,0.75826448,0.11157025,0.,0.08057851,0.0785124,0.0661157,0.04958678,0.04752066,0.03512397,0.03099173,0.02892562,0.03512397,0.0268595,0.0392562,0.03512397,0.04338843,0.04752066,0.05371901,0.05371901,0.07024793,0.07231405,0.08471075,0.09710744,0.12190083,0.1322314,0.16942149,0.19628099,0.21487603,0.23553719,0.25413224,0.2644628,0.28512397,0.27272728,0.26652893,0.23966943,0.21487603,0.17355372,0.1570248,0.12396694,0.12190083,0.10743801,0.1053719,0.09710744,0.1053719,0.09917355,0.1053719,0.09917355,0.10743801,0.10743801,0.11570248,0.11157025,0.12190083,0.11157025,0.11983471,0.11157025,0.11363637,0.11157025,0.12190083,0.1053719,0.10743801,0.10123967,0.10123967,0.08677686,0.09297521,0.08471075,0.08264463,0.0785124,0.0785124,0.07024793,0.07644628,0.06818182,0.0785124,0.07024793,0.06818182,0.06818182,0.07438017,0.07231405,0.09090909,0.10123967,0.10743801,0.1053719,0.12190083,0.11570248,0.10950413,0.09710744,0.10330579,0.09710744,0.08677686,0.07231405,0.07024793,0.05371901,0.05785124,0.04958678,0.05785124,0.05165289,0.05578512,0.05371901,0.05371901,0.,0.01239669,0.18801653,0.68181819,0.97520661,0.61570245,0.04132231,0.01239669,0.08677686,0.0661157,0.0661157,0.05165289,0.0392562,0.04338843,0.03305785,0.04132231,0.03512397,0.04545455,0.04132231,0.04545455,0.04338843,0.04958678,0.04752066,0.06404959,0.06818182]) * 100
bpm = 85


'''
graph_updelay = 80
mac_address = "B8:D6:1A:6B:32:7A"
service_uuid = QBluetoothUuid(QBluetoothUuid.SerialPort)
socket = QBluetoothSocket(QBluetoothServiceInfo.RfcommProtocol)
def socket_state_changed(state):
    if state == QBluetoothSocket.UnconnectedState:
        print("Socket unconnected")
    elif state == QBluetoothSocket.ConnectingState:
        print("Socket connecting")
    elif state == QBluetoothSocket.ConnectedState:
        print("Socket connected")
socket.stateChanged.connect(socket_state_changed)

socket.connectToService(QBluetoothAddress(mac_address), service_uuid)
'''


app = QApplication(sys.argv)
win = QWidget()
win.setWindowTitle('ClearCardio')
win.resize(650, 900) 
win.setStyleSheet("background-color: rgb(255, 255, 255);")
layout = QVBoxLayout()
win.setLayout(layout)

title = QLabel('John Doe')
font = title.font()
font.setPointSize(50)
title.setFont(font)
title.setAlignment(Qt.AlignCenter)
layout.addWidget(title, 1)
infer_graph = QHBoxLayout()
pie_series = QPieSeries()
pie_series.setHoleSize(0.6)
for name, chance, color in zip(names, chances, colors):
  slice = pie_series.append(name, chance)
  slice.setColor(color)
chart = QChart()
chart.addSeries(pie_series)
chart.setAnimationOptions(QChart.SeriesAnimations)
chart.legend().setVisible(False)
chart_view = QChartView(chart)
chart_view.setRenderHint(QPainter.Antialiasing)
chart_view.setFixedSize(400, 400)
infer_graph.addWidget(chart_view, 9)

legend_layout = QVBoxLayout()
legend_layout.addStretch(1)
for slice in pie_series.slices():
  color = slice.color().name()
  square = QPixmap(10, 10)
  square.fill(slice.color())
  icon = QLabel()
  icon.setPixmap(square)
  label = QLabel(f'{slice.label()} {slice.value()}%')
  item_layout = QHBoxLayout()
  item_layout.addWidget(icon)
  item_layout.addWidget(label)
  item_layout.addStretch(1)
  legend_layout.addLayout(item_layout)
legend_layout.addStretch(1)

image_label = QLabel()
image = QPixmap("img.png")
image_label.setPixmap(image)

side_layout = QVBoxLayout()
side_layout.addWidget(image_label, alignment=Qt.AlignTop)
side_layout.addStretch(1)
side_layout.addLayout(legend_layout)

infer_graph.addLayout(side_layout, 1)
layout.addLayout(infer_graph, 6)

ecg_series = QLineSeries()
pen = QPen(QColor(110, 100, 100))
pen.setWidth(2)
ecg_series.setPen(pen)
for x, y in enumerate(ecg_data):
  ecg_series.append(x, y)
chart = QChart()
chart.legend().hide()
chart.addSeries(ecg_series)
chart.createDefaultAxes()
y_axis = QValueAxis()
y_axis.setRange(0, 100)
y_axis.setLabelsVisible(False)
y_axis.setGridLineVisible(False)
y_axis.setLineVisible(False)
chart.setAxisY(y_axis, ecg_series)
x_axis = QValueAxis()
x_axis.setLabelsVisible(False)
x_axis.setGridLineVisible(False)
x_axis.setLineVisible(False)
chart.setAxisX(x_axis, ecg_series)
chart.setTitle("LIVE ECG")
chart_view = QChartView(chart)
chart_view.setRenderHint(QPainter.Antialiasing)
layout.addWidget(chart_view, 3)

extra_info = QLabel('''
Device Name: ACE-DA02
UUID: U38D-29G3-2935-2351
Manufactured: 2023/08/20
Paired: 2023/09/05
Registered Owner: John Doe
''')
font = extra_info.font()
font.setPointSize(16)
extra_info.setFont(font)
extra_info_layout = QHBoxLayout()
extra_info_layout.addWidget(extra_info, 1)
extra_info_layout.setContentsMargins(70, 00, 00, 00)
layout.addLayout(extra_info_layout, 3)

def get_bpm(ecg_data, thresh=1000):
  ecg_data_diff = np.diff(ecg_data)
  beat_wave = ecg_data_diff * ecg_data[1:]
  masked_beat_wave = beat_wave * (beat_wave > thresh)
  masked_beat_wave_diff = np.diff(masked_beat_wave)
  beats = masked_beat_wave_diff[:-1] * masked_beat_wave_diff[1:] < 0
  beat_times = np.where(beats == 1)[0]
  if len(beat_times) > 1:
    first, last = beat_times[-2:]
    gap = (last - first) / 360
    return 60 / gap
  else:
    return None

'''
def update_ecg():
  global ecg_series, ecg_data, heart_rate
  bpm = get_bpm(ecg_data)
  if bpm:
    heart_rate.setText(f'{int(bpm)} BPM')
  for x, y in enumerate(ecg_data):
    ecg_series.replace(x, x, y)

ecg_timer = QTimer()
ecg_timer.timeout.connect(lambda: update_ecg())
ecg_timer.start(graph_updelay)

def update_infer():
  global pie_series, chances, legend_layout, normality
  normality.setText(f{chance[0]}%')
  for slice, chance in zip(pie_series.slices(), chances):
    slice.setValue(chance)
  for i, item_layout in enumerate(legend_layout.itemAt(i).layout() for i in range(1, 6)):
    label = item_layout.itemAt(1).widget()
    label.setText(f'{names[i]} {chances[i]}%')

pie_timer = QTimer()
pie_timer.timeout.connect(lambda: update_infer())
pie_timer.start(graph_updelay)

def read():
  try:
    num = eval(str(socket.readLine(), "utf-8"))
    ecg_data[:-1] = ecg_data[1:]
    ecg_data[-1] = num
  except:
    pass

read_timer = QTimer()
read_timer.timeout.connect(lambda: read())
read_timer.start(1)
'''

'''
class Receiver(QThread):

  def __init__(self):
    super().__init__()

  def run(self):
    global ecg_data
    while 1:
      try:
        num = eval(str(socket.readLine(), "utf-8"))
        ecg_data[:-1] = ecg_data[1:]
        ecg_data[-1] = num
        print(num)
      except:
        print('failed')
        pass

blue_thread = Receiver()
blue_thread.start()
'''


def plot_sect(res, vals, max_val=1):
  max_x, max_y = res
  val_len = vals.shape[0]
  x = np.arange(val_len) / (val_len - 1) * (max_x - 1)
  y = vals / max_val * (max_y - 1)
  points = np.stack((x, y), axis=-1).astype(np.int64)
  sects = np.stack((points[:-1], points[1:]), axis=-2)
  return sects

def draw_line(max_range, points):
  point_dif = np.diff(points, axis=0)[0]
  swap_axes = abs(point_dif[1]) > point_dif[0]
  if swap_axes:
    points = np.flip(points, axis=-1)
    point_dif = np.flip(point_dif)
  slope = point_dif[1] / point_dif[0]
  y_int = points[0][1] - slope * points[0][0]
  sign = -1 if point_dif[0] < 0 else 1
  x = max_range[points[0][0]: points[1][0]: sign]
  y = (x * slope + y_int).astype(np.int64)
  if swap_axes:
    return y, x
  return x, y
v_draw_line = np.vectorize(draw_line, signature='(a),(b,c)->(),()', otypes=[np.ndarray, np.ndarray])

def graph(res, vals, max_val=1):
  max_range = np.arange(res[0] if res[0] > res[1] else res[1], dtype=np.int64)
  points = plot_sect(res, vals, max_val)
  x, y = v_draw_line(max_range, points)
  x = np.hstack(x)
  y = np.hstack(y)
  canvas = np.zeros(res, bool)
  canvas[x, y] = True
  canvas[*points[-1, -1]] = True
  return canvas

class Predictor(QThread):

  def __init__(self):
    super().__init__()
    model_file = 'ace.ckpt'
    self.cnn = model.CNN()
    self.cnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    self.cnn.eval()

  def run(self):
    global ecg_data, chances
    while 1:
      inputs = torch.tensor(ecg_data[-320:])
      inputs = torch.nn.functional.avg_pool1d(inputs.unsqueeze(0), kernel_size=12, stride=1)[0].float()
      inputs -= inputs.min()
      in_max = inputs.max().float()
      inputs /= in_max if in_max else 1.
      image = torch.tensor(graph((256, 256), inputs)).float()
      with torch.no_grad():
        logits = self.cnn(image.unsqueeze(0).unsqueeze(0))
      probabilities = torch.nn.functional.softmax(logits)[0].numpy()
      for i, probability in enumerate(probabilities):
        chances[i] = int(probability * 10000)/100

'''
ai_thread = Predictor()
ai_thread.start()
'''



center = QWidget()
center.setStyleSheet("background-color: transparent; color: black;")
text_layout = QVBoxLayout()
center.setLayout(text_layout)
normality = QLabel("91%")
font = normality.font()
font.setPointSize(40)
normality.setFont(font)
heart_rate = QLabel("85 BPM")
text_layout.addStretch(1)
text_layout.addWidget(normality, alignment=Qt.AlignCenter)
text_layout.addWidget(heart_rate, alignment=Qt.AlignCenter)
text_layout.addStretch(1)
center.setParent(win)
win.show()
center.setGeometry(chart_view.x(), chart_view.y() - 400, 400, 400)
app.exec_()
