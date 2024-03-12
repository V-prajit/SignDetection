import sys 
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLineEdit, QLabel, QVBoxLayout, QSlider, QWidget
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen

class InteractiveVideoWidget(QVideoWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.startPoint = QPoint()
        self.endPoint = QPoint()
        self.isDrawing = False
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
        print(f"Crop Rectangle: {self.startPoint}, {self.endPoint}")

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

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100,100, 1280, 720)

        videoWidget = InteractiveVideoWidget()

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

        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        layout.addWidget(videoWidget)
        layout.addWidget(self.slider)
        layout.addWidget(self.loadPlayButton)
        layout.addWidget(self.setStartButton)
        layout.addWidget(self.setEndButton)
        layout.addWidget(self.trimButton)

        self.mediaPlayer.setVideoOutput(videoWidget)

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
        print(f"Start Time Set: {self.startTime} ms")

    def setEndTime(self):
        self.endTime = self.mediaPlayer.position()
        self.trimButton.setEnabled(True)  # Ensure trim button is enabled
        print(f"End Time Set: {self.endTime} ms")

    def trimVideo(self):
        print(f"Trimming video from {self.startTime} ms to {self.endTime} ms")

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

