import glob
import json
import os.path
from typing import *
from typing import List

import numpy as np
import skimage.io as io
import torch
from torch import Tensor

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from yacs.config import CfgNode

import bism.eval.run

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, frameon=False, layout='constrained')
        self.axes = fig.add_subplot(111)
        self.axes.use_sticky_edges = True

        super(MplCanvas, self).__init__(fig)

        self.setMaximumHeight(150)
        self.setMinimumHeight(10)

class ModelCard(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super(ModelCard, self).__init__()

        self.tree = QTreeWidget()
        self.plot = MplCanvas(self, width=50, height=50, dpi=100)
        self.file = None

        self.p = None

        self.run_model_button = QPushButton('Run Model')
        self.run_model_button.clicked.connect(self.run_model)

        self.text = QPlainTextEdit()
        self.text.setMaximumHeight(150)

        self.show()

        group = QGroupBox()
        layout = QVBoxLayout()
        layout.addWidget(self.plot)
        layout.addWidget(self.tree)
        layout.addWidget(self.run_model_button)
        layout.addWidget(self.text)
        group.setLayout(layout)

        self.setLayout(layout)

    def run_model(self):
        # launch file select widget
        if self.p is None:
            file_path, _ = QFileDialog.getOpenFileName(self, 'Select File')
            path, name = os.path.split(file_path)

            self.p = QProcess()
            self.p.readyReadStandardOutput.connect(self.handle_stdout)
            self.p.readyReadStandardError.connect(self.handle_stderr)
            self.p.stateChanged.connect(self.handle_state)
            self.p.finished.connect(self.process_finished)  # Clean up
            self.p.start('python', ['../eval', '-i', file_path, '-m', self.file, '--log', '2'])

    def message(self, s):
        self.text.appendPlainText(s)

    def handle_stderr(self):
        data = self.p.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.message(stderr)

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.message(stdout)

    def handle_state(self, state):
        states = {
            QProcess.NotRunning: 'Not running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Running',
        }
        state_name = states[state]
        self.message(f"State changed: {state_name}")

    def process_finished(self):
        self.message("Process finished.")
        self.p = None

    def fill_item(self, item, value):
        item.setExpanded(False)
        if type(value) is dict:
            for key, val in sorted(value.items()):
                child = QTreeWidgetItem()
                child.setText(0, str(key))
                item.addChild(child)
                self.fill_item(child, val)
        elif type(value) is list:
            for val in value:
                child = QTreeWidgetItem()
                item.addChild(child)
                if type(val) is dict:
                    child.setText(0, '[dict]')
                    self.fill_item(child, val)
                elif type(val) is list:
                    child.setText(0, '[list]')
                    self.fill_item(child, val)
                else:
                    child.setText(0, str(val))
                child.setExpanded(False)
        else:
            child = QTreeWidgetItem()
            child.setText(0, str(value))
            item.addChild(child)

    def fill_widget(self, widget, value):
        widget.clear()
        self.fill_item(widget.invisibleRootItem(), value)

    def convert_to_dict(self, cfg_node, key_list):
        if not isinstance(cfg_node, CfgNode):
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = self.convert_to_dict(v, key_list + [k])
            return cfg_dict

    def setFile(self, file: str):
        self.file=file

    def setTextFromCfg(self, cfg: CfgNode):
        dict = self.convert_to_dict(cfg, [])
        self.fill_widget(self.tree, dict)

    def plotLoss(self, loss: List[float]):

        ax = self.plot.axes
        num_epoch = len(loss)
        epocs = list(range(num_epoch))
        ax.semilogy(epocs, loss)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')


class MainApplication(QMainWindow):
    def __init__(self):
        super(MainApplication, self).__init__()
        # self.card_container = CardContainer()

        # self.setCentralWidget(self.card_container)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.endswith('.trch'):
                self.createCard(f)
    def createCard(self, path: str | None = None):
        if path is None:
            return

        sd = torch.load(path, map_location='cpu')
        if 'cfg' in sd:
            cfg = sd['cfg']

            widget = ModelCard()
            filename=os.path.split(path)[1]

            left_dock = QDockWidget(filename, self)
            # left_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
            left_dock.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

            left_dock.setWidget(widget)
            self.addDockWidget(Qt.LeftDockWidgetArea, left_dock, Qt.Horizontal)

            widget.setFile(path)
            widget.setTextFromCfg(cfg)
            widget.plotLoss(sd['avg_epoch_loss'])
            # widget.plotLoss(sd['avg_val_loss'])




if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    application = MainApplication()
    version = 'alpha'

    geometry = app.primaryScreen().geometry()
    w, h = geometry.width(), geometry.height()
    x = (w - application.width()) / 2
    y = (h - application.height()) / 2
    application.move(x, y)
    application.setWindowTitle(f'bism model inspector - {version}')
    application.show()

    sys.exit(app.exec())
