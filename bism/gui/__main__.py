from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication
from bism.gui.app import MainApplication

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    application = MainApplication()
    application.resize(QSize(300, 200))
    application.setMinimumHeight(200)
    version = 'alpha'

    geometry = app.primaryScreen().geometry()
    w, h = geometry.width(), geometry.height()
    x = (w - application.width()) / 2
    y = (h - application.height()) / 2
    application.move(x, y)
    application.setWindowTitle(f'bism model inspector - {version}')
    application.show()

    sys.exit(app.exec())
