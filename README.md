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
- GPU acceleration for faster processing
- Parallel processing for database population

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
  - psutil (for monitoring)
  - matplotlib (for visualization)

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
   pip install PyQt6 opencv-python numpy ffmpeg-python py4j mediapipe scipy psutil matplotlib
   ```

4. For GPU acceleration:
   - For NVIDIA GPUs: Install CUDA and the CUDA-enabled version of OpenCV
     ```bash
     pip install opencv-python-contrib
     ```
   - For Apple Silicon (M1/M2): Install tensorflow-metal (optional, for additional acceleration)
     ```bash
     pip install tensorflow-metal
     ```

5. Compile and run the Java DTW implementation:
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
2. Generate a list of videos to process:
   ```bash
   python generate_video_list.py /path/to/videos --one-handed
   ```
   - Use `--one-handed` flag if videos are mostly one-handed signs
   - Review and edit the generated `videos_to_add.txt` file as needed

3. Run the enhanced database populator with GPU acceleration:
   ```bash
   python DatabasePopulator.py --workers 4 --batch_size 10 --video_list videos_to_add.txt
   ```
   - Adjust the `--workers` parameter to match your system (usually CPU core count minus 1)
   - The `--batch_size` parameter controls how many videos to process before saving

4. Monitor resource usage during processing (optional):
   ```bash
   python resource_monitor.py --output resource_log.csv
   ```

## Performance Optimization

### Hardware-Specific Optimizations

#### NVIDIA GPUs (e.g., GTX 1660)
- The system automatically detects and uses CUDA acceleration for video processing
- MediaPipe hand tracking utilizes GPU acceleration
- FFmpeg operations are hardware-accelerated using CUDA
- For best performance, use 3-4 worker processes (adjust based on your RAM)

#### Apple Silicon (M1/M2 Macs)
- Video processing uses VideoToolbox hardware acceleration
- MediaPipe hand tracking benefits from Metal GPU acceleration
- Optimized memory usage for Apple's unified memory architecture
- Best performance with 4-6 worker processes on M1 Pro/Max

#### CPU-Only Systems
- The system falls back to multi-threaded CPU processing
- FFmpeg operations use multiple CPU threads
- Recommended to use (CPU core count - 1) worker processes

### Resource Monitoring

The included `resource_monitor.py` script provides real-time monitoring of:
- CPU usage
- Memory consumption
- GPU utilization (on supported platforms)
- Process count

It generates CSV logs and visual graphs to help optimize worker count for your specific hardware.

## How It Works

1. **Video Processing**: The system extracts the specified segment of the video and crops it to the region of interest, using hardware acceleration when available.
2. **Face Detection**: Detects the face to establish a reference point for normalization.
3. **Hand Tracking**: Tracks hand positions throughout the video using MediaPipe with GPU acceleration.
4. **Feature Extraction**: Extracts features like hand centroids, orientations, and hand shape descriptors.
5. **DTW Matching**: Uses Dynamic Time Warping to compare the extracted features with signs in the database.
6. **Ranking**: Ranks potential matches based on similarity scores.

## Project Structure

- **app.py**: Main GUI application (lightweight, runs on Raspberry Pi).
- **sign_matcher.py**: Implements sign comparison logic using DTW.
- **database_manager.py**: Handles sign database operations.
- **VideoTrimAndCropping.py**: Video processing functions with hardware acceleration.
- **HandCoordinates.py**: GPU-accelerated hand tracking and feature extraction.
- **faceDetection.py**: Face detection for normalization.
- **DTW.java/FastDTW.java**: Java implementation of the DTW algorithm.
- **DTWServer.java**: Java server for DTW calculations.
- **DatabasePopulator.py**: Multi-process database population with GPU acceleration.
- **resource_monitor.py**: System resource monitoring and visualization.
- **sign_database/**: Directory containing sign data.

## Benchmarking

Run the benchmark script to evaluate system performance:
```bash
python benchmark.py
```
The results will be saved in `benchmark_results.json` and can be used to tune parameters.

## Troubleshooting

### GPU Not Being Used
- Verify proper GPU drivers are installed
- Check that OpenCV has GPU/CUDA support
- Run the resource monitor to confirm GPU activity
- Increase batch size to ensure GPU is fully utilized

### Process Crashes with Out-of-Memory Errors
- Reduce the number of worker processes
- Decrease batch size
- Ensure you have sufficient system RAM (16GB+ recommended)

### Slow Processing Despite GPU
- Check hardware acceleration with `ffmpeg -hwaccels`
- Ensure MediaPipe is utilizing the GPU
- Verify the Java DTW server is running
- Use resource monitor to identify bottlenecks