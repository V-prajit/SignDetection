#Main BackEnd File

import ffmpeg
import os
from faceDetection import detect_face
from HandCoordinates import HandCoordinates
import numpy as np
from LinearInterpolation import InterpolateAndResample, calculate_unit_vector

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
        #Get the HandCoordinates Of The Video
        centroids_dom_arr, centroids_nondom_arr, origin, l_delta_arr = HandCoordinates(videoDir, origin, scaling_factor, isOneHanded)
        #Linearly interpolate the data
        Interpolated_Dominant_Hand = InterpolateAndResample(centroids_dom_arr)
        Interpolated_nonDominant_Hand = InterpolateAndResample(centroids_nondom_arr)
        Interpolated_l_Delta = InterpolateAndResample(l_delta_arr)
        o_d = calculate_unit_vector(Interpolated_Dominant_Hand)
        o_nd = calculate_unit_vector(Interpolated_nonDominant_Hand) if not isOneHanded else np.full_like(centroids_dom_arr, np.nan)
        o_delta = calculate_unit_vector(Interpolated_l_Delta) if not isOneHanded else np.full_like(centroids_dom_arr, np.nan)

        print("Resampled Dominant Hand Data:")
        print(Interpolated_Dominant_Hand)
        print("Resampled Non-Dominant Hand Data:")
        print(Interpolated_nonDominant_Hand)
        print("Resampled l_Delta Data:")
        print(Interpolated_l_Delta)
        print("Unit Vector o_d:")
        print(o_d)
        print("Unit Vector o_nd:")
        print(o_nd)
        print("Unit Vector o_delta:")
        print(o_delta)

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        print(f"Failed to process video: {fileName}")

