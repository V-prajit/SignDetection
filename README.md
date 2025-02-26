# SignDetection

A dynamic sign language detection and recognition system built with Python, using the DTW (Dynamic Time Warping) algorithm to match sign language motions captured from videos.

## Overview

SignDetection is a tool that enables the comparison and matching of sign language gestures from video inputs. The system tracks hand movements, normalizes the data, and uses Dynamic Time Warping (DTW) to find the closest matching sign from a database of known signs.

This project is being developed under the guidance of **Professor Vassilis Athitsos** and implements the methods described in the paper:

> **A System for Large Vocabulary Sign Search**  
> *Haijing Wang, Alexandra Stefan, Sajjad Moradi, Vassilis Athitsos, Carol Neidle, and Farhad Kamangar*  
> *Computer Science and Engineering Department, University of Texas at Arlington*  

The system builds upon prior research to enable the lookup of unknown sign language gestures using video input, leveraging DTW-based trajectory comparison combined with hand appearance analysis.

### Key Features:
- Video capture and processing with precise timing control
- Face detection for motion normalization
- Hand tracking using computer vision
- DTW algorithm implementation for comparing sign patterns
- PyQt6-based GUI for easy interaction
- Database management for storing and retrieving sign data

## Requirements

- Python 3.9+
- Required Python packages:
  - PyQt6
  - opencv-python
  - numpy
  - ffmpeg-python
  - py4j
  - mediapipe
  - scipy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SignDetection.git
   cd SignDetection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install PyQt6 opencv-python numpy ffmpeg-python py4j mediapipe scipy
   ```

4. Compile and run the Java DTW implementation:
   ```bash
   javac DTW.java FastDTW.java DTWServer.java
   java DTWServer
   ```

## Usage

### Running the Application

1. Start the DTW server:
   ```bash
   java DTWServer
   ```

2. Launch the application:
   ```bash
   python app.py
   ```

### Using the GUI

1. **Load Video**: Click "Load Video" to select a sign language video file.
2. **Set Time Range**: Use "Set Start Time" and "Set End Time" to define the segment containing the sign.
3. **Select Region**: Draw a rectangle around the signing area by clicking and dragging.
4. **One-Handed Option**: Check the "One-Handed Video" box if the sign only uses one hand.
5. **Process Video**: Click "Process Video" to analyze the sign.
6. **View Results**: The system will display matching signs from the database with similarity scores.

### Adding New Signs to the Database

To add a new sign:
1. Prepare a clear video of the sign.
2. Use the application to select the time range and region.
3. Run:
   ```bash
   python DatabasePopulator.py
   ```
   with appropriate parameters.

## How It Works

1. **Video Processing**: The system extracts the specified segment of the video and crops it to the region of interest.
2. **Face Detection**: Detects the face to establish a reference point for normalization.
3. **Hand Tracking**: Tracks hand positions throughout the video.
4. **Feature Extraction**: Extracts features like hand centroids, orientations, and hand shape descriptors.
5. **DTW Matching**: Uses Dynamic Time Warping to compare the extracted features with signs in the database.
6. **Ranking**: Ranks potential matches based on similarity scores.

## Project Structure

- **app.py**: Main GUI application.
- **sign_matcher.py**: Implements sign comparison logic using DTW.
- **database_manager.py**: Handles sign database operations.
- **VideoTrimAndCropping.py**: Video processing functions.
- **HandCoordinates.py**: Hand tracking and feature extraction.
- **faceDetection.py**: Face detection for normalization.
- **DTW.java/FastDTW.java**: Java implementation of the DTW algorithm.
- **DTWServer.java**: Java server for DTW calculations.
- **sign_database/**: Directory containing sign data.

## Benchmarking

Run the benchmark script to evaluate system performance:
```bash
python benchmark.py
```
The results will be saved in `benchmark_results.json` and can be used to tune parameters.