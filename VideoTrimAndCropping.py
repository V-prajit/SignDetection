import ffmpeg
import os
import json
from faceDetection import detect_face
from HandCoordinates import HandCoordinates
import numpy as np
from LinearInterpolation import InterpolateAndResample
from sign_matcher import SignMatcher
import cv2
from hand_processing import extract_hand_image, preprocess_hand_image

def load_database(db_file="sign_database.json"):
    if os.path.exists(db_file):
        with open(db_file, 'r') as f:
            return json.load(f)["signs"]
    return {}

def GetValues(startTime, endTime, startPoint, endPoint, fileName, isOneHanded, add_to_db=False):
    print(f"Processing video: {fileName}")
    print(f"Time range: {startTime} to {endTime}")
    print(f"ROI points: {startPoint} to {endPoint}")
    
    start_seconds = startTime / 1000.0
    end_seconds = endTime / 1000.0

    if endTime <= startTime:
        print("Error: End time must be greater than start time.")
        return [], None, None, None

    cap = cv2.VideoCapture(fileName)
    if not cap.isOpened():
        print(f"Error: Could not open video {fileName}")
        return [], None, None, None
        
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    width = min(endPoint.x() - startPoint.x(), original_width)
    height = min(endPoint.y() - startPoint.y(), original_height)
    x = min(startPoint.x(), original_width - width)
    y = min(startPoint.y(), original_height - height)

    MIN_SIZE = 200
    if width < MIN_SIZE or height < MIN_SIZE:
        width = max(MIN_SIZE, width)
        height = max(MIN_SIZE, height)

    baseName = os.path.basename(fileName)
    fileName_NoExtension, extension = os.path.splitext(baseName)
    output_fileName = f"{fileName_NoExtension}_transformed{extension}"

    crop_dimensions = f'{width}:{height}:{x}:{y}'
    print(f"Crop dimensions: {crop_dimensions}")

    try:
        (
            ffmpeg
            .input(fileName, ss=start_seconds, t=end_seconds-start_seconds)
            .filter('crop', *crop_dimensions.split(':'))
            .output(output_fileName)
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Processed video saved as: {output_fileName}")

        origin, scaling_factor, videoDir = detect_face(output_fileName)
        if origin is None or scaling_factor is None:
            print("Using default normalization parameters")
            origin = (width/2, height/2)
            scaling_factor = 1.0/height
            videoDir = output_fileName

        print(f"Face detection parameters - Origin: {origin}, Scaling: {scaling_factor}")

        (centroids_dom_arr,
         centroids_nondom_arr,
         hand_boxes_dom,
         hand_boxes_nondom,
         origin,
         l_delta_arr,
         orientation_dom_arr,
         orientation_nondom_arr,
         orientation_delta_arr) = HandCoordinates(videoDir, origin, scaling_factor, isOneHanded)

        cap = cv2.VideoCapture(videoDir)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            return [], None, None, None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        if not ret:
            print("Failed to read last frame")
            return [], None, None, None
        
        cap.release()

        dom_start_img = extract_hand_image(first_frame, hand_boxes_dom[0])
        dom_end_img = extract_hand_image(last_frame, hand_boxes_dom[-1])
        
        H_d_s = preprocess_hand_image(dom_start_img)
        H_d_e = preprocess_hand_image(dom_end_img)

        TARGET_FRAMES = 20
        Interpolated_Dominant_Hand = InterpolateAndResample(centroids_dom_arr, TARGET_FRAMES)
        Interpolated_nonDominant_Hand = InterpolateAndResample(centroids_nondom_arr, TARGET_FRAMES)
        Interpolated_l_Delta = InterpolateAndResample(l_delta_arr, TARGET_FRAMES)
        Interpolated_orient_dom = InterpolateAndResample(orientation_dom_arr, TARGET_FRAMES)
        Interpolated_orient_nondom = InterpolateAndResample(orientation_nondom_arr, TARGET_FRAMES)
        Interpolated_orient_delta = InterpolateAndResample(orientation_delta_arr, TARGET_FRAMES)

        processed_features = {
            'centroids_dom_arr': Interpolated_Dominant_Hand.tolist(),
            'centroids_nondom_arr': Interpolated_nonDominant_Hand.tolist(),
            'l_delta_arr': Interpolated_l_Delta.tolist(),
            'orientation_dom_arr': Interpolated_orient_dom.tolist(),
            'orientation_nondom_arr': Interpolated_orient_nondom.tolist(),
            'orientation_delta_arr': Interpolated_orient_delta.tolist(),
            'H_d_s': H_d_s,
            'H_d_e': H_d_e,
            'is_one_handed': isOneHanded,
            'frame_count': TARGET_FRAMES
        }

        if not isOneHanded and len(hand_boxes_nondom) > 0:
            nondom_start_img = extract_hand_image(first_frame, hand_boxes_nondom[0])
            nondom_end_img = extract_hand_image(last_frame, hand_boxes_nondom[-1])
            
            H_nd_s = preprocess_hand_image(nondom_start_img)
            H_nd_e = preprocess_hand_image(nondom_end_img)
            
            processed_features.update({
                'H_nd_s': H_nd_s,
                'H_nd_e': H_nd_e
            })

        if add_to_db:
            db_data = load_database()
            db_data[fileName] = {
                "name": os.path.splitext(os.path.basename(fileName))[0],
                "features": processed_features,
                "is_one_handed": isOneHanded,
                "duration": float(end_seconds - start_seconds),
                "origin": [float(x) for x in origin] if isinstance(origin, (tuple, list)) else [0.0, 0.0],
                "scaling_factor": float(scaling_factor)
            }
            with open("sign_database.json", 'w') as f:
                json.dump({"signs": db_data}, f, indent=4)
            print(f"Added sign data to database: {fileName}")

        matches = []
        db_data = load_database()

        if db_data:
            matcher = SignMatcher()
            
            database_signs = []
            sign_names = []
            
            for path, entry in db_data.items():
                if entry['is_one_handed'] != isOneHanded:
                    continue
                database_signs.append(entry['features'])
                sign_names.append(entry['name'])
            
            distance_matches = matcher.find_matches(processed_features, database_signs, top_k=10)
            
            for idx, similarity in distance_matches:
                sign_name = sign_names[idx]
                matches.append((sign_name, similarity))
                print(f"Match: {sign_name}, Similarity: {similarity:.2f}%")

        return matches, origin, scaling_factor, processed_features

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode() if e.stderr else str(e))
        print(f"Failed to process video: {fileName}")
        return [], None, None, None
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], None, None, None