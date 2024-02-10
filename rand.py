import sys 
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QStackedWidget, QHBoxLayout, QFrame
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl, QRectF, Qt, QEvent
from PyQt6.QtGui import QMouseEvent, QPen, QColor, QBrush

class SimpleGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.scene = scene
        self.drawing = False
        self.currentRect = None
        self.startPos = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.startPos = self.mapToScene(event.pos())
            self.currentRect = QGraphicsRectItem(QRectF(self.startPos, self.startPos))
            pen = QPen(QColor(255, 0, 0))  # Red color for the pen
            pen.setWidth(3)  # Set the width of the pen
            self.currentRect.setPen(pen)
            brush = QBrush(QColor(0, 255, 0, 128))  # Semi-transparent green for the brush
            self.currentRect.setBrush(brush)
            self.scene.addItem(self.currentRect)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and self.currentRect:
            self.endPos = self.mapToScene(event.pos())
            rect = QRectF(self.startPos, self.endPos).normalized()
            self.currentRect.setRect(rect)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.drawing = False
        if self.currentRect:
            self.currentRect = None  # Reset currentRect to ensure drawing starts fresh next time

class SimpleViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Viewer with Video")
        self.setGeometry(100, 100, 800, 600)  # Adjust size to accommodate video

        self.layout = QVBoxLayout(self)

        # Initialize the MediaPlayer and VideoWidget
        self.mediaPlayer = QMediaPlayer()
        self.videoWidget = QVideoWidget()

        # Add VideoWidget to the layout
        self.layout.addWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # Setup scene and graphics view for drawing
        scene = QGraphicsScene(self)
        scene.setSceneRect(0, 0, 800, 600)  # Match the size to the video widget
        self.graphicsView = SimpleGraphicsView(scene, self)
        self.graphicsView.setGeometry(0, 0, 800, 600)  # Overlay graphics view on the video widget
        self.layout.addWidget(self.graphicsView)

        # Button to open video files
        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.openFile)
        self.layout.addWidget(self.openButton)

        self.mediaPlayer.errorOccurred.connect(self.handleError)
        self.mediaPlayer.stateChanged.connect(self.handleStateChanged)

    def handleError(self, error):
        print(f"Error occurred: {self.mediaPlayer.errorString()}")

    def handleStateChanged(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            print("Playback started")
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            print("Playback stopped")
        elif state == QMediaPlayer.PlaybackState.PausedState:
            print("Playback paused")

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        
        if fileName:
            print(f"Opening video file: {fileName}")  # Debugging: Confirm the file path
            self.mediaPlayer.setSource(QUrl.fromLocalFile(fileName))
            self.mediaPlayer.play()

def main():
    app = QApplication(sys.argv)
    viewer = SimpleViewer()
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
