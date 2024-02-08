import sys 
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl

class VideoPlayer (QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL App")
        self.setGeometry(100, 100, 640, 480)

        layout = QVBoxLayout()

        self.videoWidget = QVideoWidget()
        layout.addWidget(self.videoWidget)

        self.mediaPlayer = QMediaPlayer()
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.openButton = QPushButton("Open Video")
        self.openButton.clicked.connect(self.openFile)
        layout.addWidget(self.openButton)

        self.pauseButton = QPushButton("Pause")
        self.pauseButton.clicked.connect(self.togglePause)
        layout.addWidget(self.pauseButton)

        self.setLayout(layout)

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

    
