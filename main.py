import sys 
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QStackedWidget, QHBoxLayout, QFrame
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, QRectF, Qt, QEvent
from PyQt6.QtGui import QMouseEvent, QPen, QColor, QBrush

class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene, parent = None):
        super().__init__(scene, parent)
        self.scene = scene
        self.startPos = None
        self.endPos = None
        self.drawing = False
        self.currentRect = None
        self.setFrameShape(QFrame.Shape.NoFrame)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.startPos = self.mapToScene(event.pos())
            self.drawing = True
            self.currentRect = QGraphicsRectItem(QRectF(self.startPos, self.startPos))
            pen = QPen(QColor(255,0,0))
            pen.setWidth(5)
            self.currentRect.setPen(pen)
            brush = QBrush(QColor(0,255,0,255))
            self.currentRect.setBrush(brush)
            self.scene.addItem(self.currentRect)
            print("mouseClicked")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and self.currentRect:
            self.endPos = self.mapToScene(event.pos())
            rect = QRectF(self.startPos, self.endPos).normalized()
            self.currentRect.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.drawing:
            self.drawing = False
            print(f"Rectangle drawn: {self.currentRect.rect()}")
        super().mouseReleaseEvent(event)

    def eventFilter(self, source, event):
        if event.type() in [QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease]:
            # Directly call the mouse event handlers
            if event.type() == QEvent.Type.MouseButtonPress:
                self.mousePressEvent(event)
            elif event.type() == QEvent.Type.MouseMove:
                self.mouseMoveEvent(event)
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self.mouseReleaseEvent(event)
            return True
        return super().eventFilter(source, event)


class VideoPlayer (QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL App")
        self.setGeometry(100, 100, 640, 480)

        self.layout = QVBoxLayout()

        self.videoLayout = QHBoxLayout()
        self.videoWidget = QVideoWidget()

        self.stackWidget = QStackedWidget()
        self.stackWidget.addWidget(self.videoWidget)

        self.scene = QGraphicsScene()
        self.customView = CustomGraphicsView(self.scene, self)
        self.customView.setFrameShape(QFrame.Shape.NoFrame)
        self.customView.setStyleSheet("background: transparent;")
        self.stackWidget.addWidget(self.customView)

        self.videoLayout.addWidget(self.stackWidget)
        self.layout.addLayout(self.videoLayout)
        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.openFile)
        self.layout.addWidget(self.openButton)

        self.pauseButton = QPushButton("Pause")
        self.pauseButton.clicked.connect(self.togglePause)
        self.layout.addWidget(self.pauseButton)

        self.setLayout(self.layout)

        self.videoWidget.installEventFilter(self.customView)


    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")

        if fileName != '':
            self.mediaPlayer.setSource(QUrl.fromLocalFile(fileName))
            self.mediaPlayer.play()
    
    def togglePause(self):
        if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mediaPlayer.pause()
            self.pauseButton.setText("Play")
        else :
            self.mediaPlayer.play()
            self.pauseButton.setText("Pause")

def main():
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()