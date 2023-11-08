import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QPixmap
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice, QLineSeries
from PyQt5.QtCore import Qt, QMargins


names = ['Non-ecotopic beats', 'Supraventricular ectopic beats', 'Ventricular ectopic beats', 'Fusion Beats', 'Unknown Beats']
chances = [50, 20, 15, 10, 5]
colors = [QColor(100, 255, 100), QColor(255, 255, 200), QColor(255, 200, 150), QColor(255, 130, 130), QColor(200, 80, 255)]
ecg = [(0, 0), (1, 10), (2, 30), (4, 5), (10, 20)]


app = QApplication(sys.argv)
win = QWidget()
win.setWindowTitle('ClearCardio')
win.resize(500, 800) 
win.setStyleSheet("background-color: rgb(255, 255, 255);")
layout = QVBoxLayout()
win.setLayout(layout)

title = QLabel('AI Prediction of Heart Status')
font = title.font()
font.setPointSize(16)
title.setFont(font)
title.setAlignment(Qt.AlignCenter)
layout.addWidget(title, 1)
infer_graph = QVBoxLayout()
series = QPieSeries()
series.setHoleSize(0.5)
for name, chance, color in zip(names, chances, colors):
  slice = series.append(name, chance)
  slice.setColor(color)
chart = QChart()
chart.setMargins(QMargins(0, 0, 0, 0));
chart.addSeries(series)
chart.setAnimationOptions(QChart.SeriesAnimations)
chart.legend().setVisible(False)
chart_view = QChartView(chart)
chart_view.setRenderHint(QPainter.Antialiasing)
infer_graph.addWidget(chart_view, 9)

legend_layout = QVBoxLayout()
legend_layout.addStretch(1)
for slice in series.slices():
  color = slice.color().name()
  square = QPixmap(10, 10)
  square.fill(slice.color())
  icon = QLabel()
  icon.setPixmap(square)
  label = QLabel(slice.label())
  share = QLabel(f'{slice.value()}%')
  item_layout = QHBoxLayout()
  item_layout.addWidget(share)
  item_layout.addWidget(icon)
  item_layout.addWidget(label)
  item_layout.addStretch(1)
  legend_layout.addLayout(item_layout)
legend_layout.addStretch(1)
infer_graph.addLayout(legend_layout, 1)
layout.addLayout(infer_graph, 6)

series = QLineSeries()
for point in ecg:
  series.append(*point)
chart = QChart()
chart.legend().hide()
chart.addSeries(series)
chart.createDefaultAxes()
chart.setTitle("LIVE ECG")
chart_view = QChartView(chart)
chart_view.setRenderHint(QPainter.Antialiasing)
layout.addWidget(chart_view, 3)

win.show()
app.exec_()
