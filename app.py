import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QSlider, QWidget, QCheckBox, QListWidget,
    QLabel, QStackedWidget, QGraphicsScene, QGraphicsView
)
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtCore import Qt, QUrl, QRectF, QPointF
from PyQt6.QtGui import QPen, QColor
from VideoTrimAndCropping import GetValues
from database_manager import SignDatabase
import cv2
class VideoGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        
        self.startPoint = None
        self.currentRect = None
        self.isDrawing = False

        self.scene.setSceneRect(QRectF(0, 0, 640, 480))
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.video_item.setSize(self.scene.sceneRect().size())

    def mousePressEvent(self, event):
        self.startPoint = self.mapToScene(event.pos())
        self.isDrawing = True
        if self.currentRect:
            self.scene.removeItem(self.currentRect)
        self.currentRect = self.scene.addRect(
            QRectF(self.startPoint, self.startPoint),
            QPen(QColor(255, 0, 0), 2)
        )

    def mouseMoveEvent(self, event):
        if self.isDrawing and self.currentRect:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.startPoint, current_point).normalized()
            self.currentRect.setRect(rect)

    def mouseReleaseEvent(self, event):
        self.isDrawing = False
        if self.currentRect:
            self.endPoint = self.mapToScene(event.pos())

    def get_selection_points(self):
        if self.currentRect:
            rect = self.currentRect.rect()
            start = self.mapFromScene(rect.topLeft())
            end = self.mapFromScene(rect.bottomRight())
            return start, end
        return None, None

class VideoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "ASL"
        self.mediaPlayer = QMediaPlayer(self)
        self.videoLoaded = False
        self.startTime = 0
        self.endTime = 0
        self.startTimeSet = False
        self.endTimeSet = False
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
        
        self.sign_db = SignDatabase(data_dir="sign_database")
        print(f"Initialized sign database from: sign_database")
        
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1280, 720)

    def setup_video_page(self):
        layout = QVBoxLayout(self.video_page)
        
        self.videoWidget = VideoGraphicsView()
        layout.addWidget(self.videoWidget)
        
        self.mediaPlayer.setVideoOutput(self.videoWidget.video_item)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
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
                
                self.trimButton.setEnabled(True)
                
                self.playVideo()

    def playVideo(self):
        self.mediaPlayer.play()
        self.loadPlayButton.setText("Pause")

    def setStartTime(self):
        self.startTime = self.mediaPlayer.position()
        self.startTimeSet = True
        self.trimButton.setEnabled(True)

    def setEndTime(self):
        self.endTime = self.mediaPlayer.position()
        self.endTimeSet = True
        self.trimButton.setEnabled(True)

    def oneHandedCheckChanged(self, state):
        self.isOneHanded = state == Qt.CheckState.Checked.value

    def trimVideo(self):
        try:
            start_point, end_point = self.videoWidget.get_selection_points()
            
            # Check if ROI has been set
            if not start_point or not end_point:
                cap = cv2.VideoCapture(self.fileName)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    start_point = QPoint(0, 0)
                    end_point = QPoint(width, height)
                    print(f"No ROI selected. Using full video dimensions: {width}x{height}")
                    cap.release()
                else:
                    print("Could not open video to get dimensions")
                    return
            
            # Check if time bounds have been set
            if self.startTime == 0 and self.endTime == 0:
                cap = cv2.VideoCapture(self.fileName)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0 and fps > 0:
                        self.endTime = (frame_count / fps) * 1000  # Convert to milliseconds
                        print(f"No time bounds set. Using full video duration: {self.endTime/1000} seconds")
                    else:
                        # Fallback duration if frame_count or fps is invalid
                        self.endTime = 5000  # 5 seconds
                        print("Warning: Could not determine video length. Using 5 seconds as default")
                    cap.release()
                else:
                    print("Could not open video to get duration")
                    return
                    
            matches, origin, scaling_factor, features = GetValues(
                self.startTime,
                self.endTime,
                start_point,
                end_point,
                self.fileName,
                self.isOneHanded
            )
                
            self.resultsList.clear()
                
            if matches:
                    for i, (sign_name, distance) in enumerate(matches, 1):
                        similarity = 1 / (1 + distance)
                        self.resultsList.addItem(
                            f"{i}. {sign_name} (Similarity: {similarity:.2f})"
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