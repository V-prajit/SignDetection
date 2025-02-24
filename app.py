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

class VideoGraphicsView(QGraphicsView):
    """
    Custom graphics view that handles video playback and rectangle drawing.
    Uses QGraphicsScene for proper layering of video and overlay graphics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create and set up the graphics scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Create video item for playback
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        
        # Initialize drawing variables
        self.startPoint = None
        self.currentRect = None
        self.isDrawing = False
        
        # Set scene size to match video dimensions
        self.scene.setSceneRect(QRectF(0, 0, 640, 480))
        
        # Configure view properties
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
    def resizeEvent(self, event):
        """Handle resize events to maintain proper video scaling"""
        super().resizeEvent(event)
        # Fit the view to the scene contents
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        # Update video item size
        self.video_item.setSize(self.scene.sceneRect().size())

    def mousePressEvent(self, event):
        """Handle mouse press to start drawing rectangle"""
        # Convert mouse coordinates to scene coordinates
        self.startPoint = self.mapToScene(event.pos())
        self.isDrawing = True
        # Create initial rectangle
        if self.currentRect:
            self.scene.removeItem(self.currentRect)
        self.currentRect = self.scene.addRect(
            QRectF(self.startPoint, self.startPoint),
            QPen(QColor(255, 0, 0), 2)
        )

    def mouseMoveEvent(self, event):
        """Handle mouse movement to update rectangle size"""
        if self.isDrawing and self.currentRect:
            # Update rectangle size based on current mouse position
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.startPoint, current_point).normalized()
            self.currentRect.setRect(rect)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish drawing rectangle"""
        self.isDrawing = False
        # Store final rectangle coordinates
        if self.currentRect:
            self.endPoint = self.mapToScene(event.pos())

    def get_selection_points(self):
        """Return the rectangle selection points in view coordinates"""
        if self.currentRect:
            rect = self.currentRect.rect()
            # Convert scene coordinates to view coordinates
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
        self.isOneHanded = False
        
        # Set up the main window
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create stacked widget for multiple pages
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.video_page = QWidget()
        self.results_page = QWidget()
        
        self.setup_video_page()
        self.setup_results_page()
        
        self.stacked_widget.addWidget(self.video_page)
        self.stacked_widget.addWidget(self.results_page)
        
        self.sign_db = SignDatabase()
        
        # Configure window
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1280, 720)

    def setup_video_page(self):
        """Set up the video playback and control page"""
        layout = QVBoxLayout(self.video_page)
        
        # Create and add video view
        self.videoWidget = VideoGraphicsView()
        layout.addWidget(self.videoWidget)
        
        # Set up media player
        self.mediaPlayer.setVideoOutput(self.videoWidget.video_item)
        
        # Create and add controls
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.setPosition)
        layout.addWidget(self.slider)
        
        # Add buttons
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
        
        # Connect media player signals
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        
        self.fileName = None

    def setup_results_page(self):
        """Set up the results display page"""
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
            start_point, end_point = self.videoWidget.get_selection_points()
            if start_point and end_point:
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