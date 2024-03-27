#Main BackEnd File

import ffmpeg
import os
from faceDetection import detect_face
from HandCoordinates import HandCoordinates
import numpy as np

def GetValues(startTime, endTime, startPoint, endPoint, fileName, isOneHanded):
    print(f"The time stamps are {startTime} and {endTime}")
    print(f"The coordinates are {startPoint} and {endPoint}")
    print(fileName)
    print(isOneHanded)
    
    # Convert startTime and endTime from milliseconds to seconds
    start_seconds = startTime / 1000.0
    end_seconds = endTime / 1000.0

    if endTime <= startTime:
        print("Error: End time must be greater than start time.")
        return

    width = endPoint.x() - startPoint.x()
    height = endPoint.y() - startPoint.y()
    x = startPoint.x()
    y = startPoint.y()

    baseName = os.path.basename(fileName)
    fileName_NoExtension, extension = baseName.rsplit('.', 1)
    output_fileName = f"{fileName_NoExtension}_transformed.{extension}"
    output_fileName_2 = f"{fileName_NoExtension}_Centroids.{extension}"

    crop_dimensions = f'{width}:{height}:{x}:{y}'

    try:
        (
            ffmpeg
            .input(fileName, ss=start_seconds, to=end_seconds)
            .filter_('crop', *crop_dimensions.split(':'))
            .output(output_fileName)
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Processed video saved as: {output_fileName}")
        origin, scaling_factor, videoDir = detect_face(output_fileName)
        print(origin, scaling_factor)
        HandCoordinates(videoDir, origin, scaling_factor, isOneHanded)

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        print(f"Failed to process video: {fileName}")

