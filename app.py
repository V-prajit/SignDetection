#Main Frontend File

import sys 
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLineEdit, QLabel, QVBoxLayout, QSlider, QWidget, QCheckBox
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QResizeEvent
from VideoTrimAndCropping import GetValues

class InteractiveVideoWidget(QVideoWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.startPoint = QPoint()
        self.endPoint = QPoint()
        self.isDrawing = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def mousePressEvent(self, event):
        self.startPoint = event.position().toPoint()
        self.endPoint = self.startPoint
        self.isDrawing = True
        self.update()

    def mouseMoveEvent(self, event):
        if self.isDrawing:
            self.endPoint = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        self.isDrawing = False
        self.endPoint = event.position().toPoint()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.startPoint.isNull() and not self.endPoint.isNull():
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
            rect = QRect(self.startPoint, self.endPoint)
            painter.drawRect(rect)



class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "ASL"
        self.mediaPlayer = QMediaPlayer(self)
        self.initUI()
        self.videoLoaded = False
        self.startTime = 0
        self.endTime = 0
        self.videoWidget.setMinimumSize(1280,720)
        self.isOneHanded = False

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100,100, 1280, 720)

        self.videoWidget = InteractiveVideoWidget()

        self.loadPlayButton = QPushButton('Load Video', self)
        self.loadPlayButton.clicked.connect(self.loadOrPlayVideo)

        self.setStartButton = QPushButton('Set Start Time', self)
        self.setStartButton.clicked.connect(self.setStartTime)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.setPosition)

        self.setEndButton = QPushButton('Set End Time', self)
        self.setEndButton.clicked.connect(self.setEndTime)

        self.trimButton = QPushButton('Trim Video', self)
        self.trimButton.clicked.connect(self.trimVideo)
        self.trimButton.setEnabled(False)

        self.oneHandedCheckBox = QCheckBox("One-Handed Video", self)
        self.oneHandedCheckBox.stateChanged.connect(self.oneHandedCheckChanged)

        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(self.videoWidget)
        layout.addWidget(self.slider)
        layout.addWidget(self.loadPlayButton)
        layout.addWidget(self.setStartButton)
        layout.addWidget(self.setEndButton)
        layout.addWidget(self.trimButton)
        layout.addWidget(self.oneHandedCheckBox)

        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.fileName = None

    def loadOrPlayVideo(self):
        if self.videoLoaded:
            # Toggle play/pause if a video is already loaded
            if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.mediaPlayer.pause()
                self.loadPlayButton.setText('Play')
            else:
                self.mediaPlayer.play()
                self.loadPlayButton.setText('Pause')
        else:
            # Load video if no video is loaded
            fileName, _ = QFileDialog.getOpenFileName(self, "Open Video File")
            if fileName:
                self.fileName = fileName
                self.mediaPlayer.setSource(QUrl.fromLocalFile(fileName))
                self.videoLoaded = True
                self.loadPlayButton.setText('Play')
                self.playVideo()

    def playVideo(self):
            self.mediaPlayer.play()
            self.loadPlayButton.setText("Pause")

    def setStartTime(self):
        self.startTime = self.mediaPlayer.position()
        self.trimButton.setEnabled(True)  # Enable trim button after setting start time

    def setEndTime(self):
        self.endTime = self.mediaPlayer.position()
        self.trimButton.setEnabled(True)  # Ensure trim button is enabled

    def oneHandedCheckChanged(self, state):
        self.isOneHanded = not self.isOneHanded
        #print("Checkbox changed:", self.isOneHanded)

    def trimVideo(self):
        GetValues(self.startTime, self.endTime, self.videoWidget.startPoint, self.videoWidget.endPoint, self.fileName, self.isOneHanded)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
    
    def durationChanged(self, duration):
        self.slider.setRange(0, duration)
    
    def positionChanged(self, position):
        self.slider.setValue(position)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoEditor()
    ex.show()
    sys.exit(app.exec())