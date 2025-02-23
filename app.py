import sys 
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, 
    QVBoxLayout, QSlider, QWidget, QCheckBox, QListWidget, 
    QLabel, QStackedWidget
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QUrl, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen
from VideoTrimAndCropping import GetValues
from database_manager import SignDatabase

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
        self.videoLoaded = False
        self.startTime = 0
        self.endTime = 0
        self.isOneHanded = False
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        
        self.video_page = QWidget()
        self.results_page = QWidget()
        
        self.setup_video_page()
        self.setup_results_page()
        
        self.stacked_widget.addWidget(self.video_page)
        self.stacked_widget.addWidget(self.results_page)
        
        self.sign_db = SignDatabase()
        
        self.setWindowTitle(self.title)
        self.setGeometry(100,100, 1280, 720)

    def setup_video_page(self):
        layout = QVBoxLayout(self.video_page)
        
        self.videoWidget = InteractiveVideoWidget()
        layout.addWidget(self.videoWidget)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,0)
        self.slider.sliderMoved.connect(self.setPosition)
        layout.addWidget(self.slider)

        self.loadPlayButton = QPushButton('Load Video')
        self.loadPlayButton.clicked.connect(self.loadOrPlayVideo)
        layout.addWidget(self.loadPlayButton)

        self.setStartButton = QPushButton('Set Start Time')
        self.setStartButton.clicked.connect(self.setStartTime)
        layout.addWidget(self.setStartButton)

        self.setEndButton = QPushButton('Set End Time')
        self.setEndButton.clicked.connect(self.setEndTime)
        layout.addWidget(self.setEndButton)

        self.trimButton = QPushButton('Process Video')
        self.trimButton.clicked.connect(self.trimVideo)
        self.trimButton.setEnabled(False)
        layout.addWidget(self.trimButton)

        self.oneHandedCheckBox = QCheckBox("One-Handed Video")
        self.oneHandedCheckBox.stateChanged.connect(self.oneHandedCheckChanged)
        layout.addWidget(self.oneHandedCheckBox)

        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.fileName = None

    def setup_results_page(self):
        layout = QVBoxLayout(self.results_page)
        
        title_label = QLabel("Matching Results")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        self.resultsList = QListWidget()
        self.resultsList.setStyleSheet("QListWidget { font-size: 14px; padding: 5px; }")
        layout.addWidget(self.resultsList)
        
        self.backButton = QPushButton("Back to Video")
        self.backButton.clicked.connect(self.show_video_page)
        layout.addWidget(self.backButton)

    def show_video_page(self):
        self.stacked_widget.setCurrentIndex(0)

    def show_results_page(self):
        self.stacked_widget.setCurrentIndex(1)

    def loadOrPlayVideo(self):
        if self.videoLoaded:
            if self.mediaPlayer.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.mediaPlayer.pause()
                self.loadPlayButton.setText('Play')
            else:
                self.mediaPlayer.play()
                self.loadPlayButton.setText('Pause')
        else:
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
        self.trimButton.setEnabled(True)

    def setEndTime(self):
        self.endTime = self.mediaPlayer.position()
        self.trimButton.setEnabled(True)

    def oneHandedCheckChanged(self, state):
        self.isOneHanded = state == Qt.CheckState.Checked.value

    def trimVideo(self):
        try:
            matches, origin, scaling_factor, features = GetValues(
                self.startTime, 
                self.endTime, 
                self.videoWidget.startPoint, 
                self.videoWidget.endPoint, 
                self.fileName, 
                self.isOneHanded
            )
            
            self.resultsList.clear()
            
            if matches:
                for i, (sign_name, distance) in enumerate(matches, 1):
                    similarity = 1 / (1 + distance)
                    self.resultsList.addItem(
                        f"{i}. {sign_name} (Similarity: {similarity:.2})"
                    )
            else:
                self.resultsList.addItem("No matching signs found")
            
            self.show_results_page()
                
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            import traceback
            traceback.print_exc()

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
    
    def durationChanged(self, duration):
        self.slider.setRange(0, duration)
    
    def positionChanged(self, position):
        self.slider.setValue(position)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = VideoEditor()
    ex.show()
    sys.exit(app.exec())