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

def get_hardware_acceleration_option():
    """Determine the best hardware acceleration option for ffmpeg on this system"""
    import platform
    import subprocess
    
    system = platform.system()
    
    # Check ffmpeg capabilities
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"], 
            capture_output=True, 
            text=True
        )
        hwaccels = result.stdout.lower()
        
        # Check for available hardware acceleration
        if "videotoolbox" in hwaccels and system == "Darwin":  # macOS (including M1/M2)
            return "videotoolbox"
        elif "cuda" in hwaccels:  # NVIDIA GPU
            return "cuda"
        elif "qsv" in hwaccels:  # Intel Quick Sync
            return "qsv"
        elif "vaapi" in hwaccels:  # Linux VA-API
            return "vaapi"
        elif "dxva2" in hwaccels and system == "Windows":  # Windows
            return "dxva2"
        elif "d3d11va" in hwaccels and system == "Windows":  # Windows
            return "d3d11va"
        else:
            return None
    except:
        return None

def load_database(db_dir="sign_database", db_file="sign_data.json"):
    db_path = os.path.join(db_dir, db_file)
    if os.path.exists(db_path):
        print(f"Loading database from: {db_path}")
        with open(db_path, 'r') as f:
            db_data = json.load(f)
            return db_data.get("signs", {})
    else:
        print(f"Database file not found at: {db_path}")
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

    # Get hardware acceleration option
    hw_accel = get_hardware_acceleration_option()
    print(f"Hardware acceleration: {hw_accel if hw_accel else 'Not available'}")

    try:
        print(f"Attempting to extract all frames from video...")
        
        # Base ffmpeg input
        input_stream = ffmpeg.input(fileName)
        
        # Apply hardware acceleration if available
        if hw_accel:
            if hw_accel == "videotoolbox":  # For macOS including M1/M2
                # Continuing from previous code
                output_stream = (
                    ffmpeg.input(fileName, hwaccel="videotoolbox")
                    .filter('fps', fps=30)
                    .output(output_fileName, vsync=0)
                    .overwrite_output()
                )
                output_stream.run(quiet=True)
            elif hw_accel == "cuda":  # For NVIDIA GPUs
                (
                    ffmpeg
                    .input(fileName, hwaccel="cuda")
                    .filter('fps', fps=30)
                    .output(output_fileName, vsync=0)
                    .overwrite_output()
                    .run(quiet=True)
                )
            elif hw_accel == "qsv":  # For Intel Quick Sync
                (
                    ffmpeg
                    .input(fileName, hwaccel="qsv")
                    .filter('fps', fps=30)
                    .output(output_fileName, vsync=0)
                    .overwrite_output()
                    .run(quiet=True)
                )
            elif hw_accel in ["vaapi", "dxva2", "d3d11va"]:  # Other acceleration methods
                (
                    ffmpeg
                    .input(fileName, hwaccel=hw_accel)
                    .filter('fps', fps=30)
                    .output(output_fileName, vsync=0)
                    .overwrite_output()
                    .run(quiet=True)
                )
        else:
            # Standard processing without hardware acceleration
            # Try with higher thread count for better CPU utilization
            (
                ffmpeg
                .input(fileName)
                .filter('fps', fps=30)
                .output(output_fileName, vsync=0, threads=8)  # Use more CPU threads
                .overwrite_output()
                .run(quiet=True)
            )
        
        # Verify the transformed video has multiple frames
        check_cap = cv2.VideoCapture(output_fileName)
        check_frame_count = int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        check_cap.release()
        print(f"Transformed video contains {check_frame_count} frames")
        
        if check_frame_count <= 1:
            # Try alternative approach with different parameters
            print("First approach failed, trying alternate method...")
            if hw_accel:
                (
                    ffmpeg
                    .input(fileName, hwaccel=hw_accel)
                    .output(output_fileName, c='copy')  # Direct stream copy with hardware accel
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                (
                    ffmpeg
                    .input(fileName)
                    .output(output_fileName, c='copy', threads=8)  # More CPU threads
                    .overwrite_output()
                    .run(quiet=True)
                )
            
            # Verify again
            check_cap = cv2.VideoCapture(output_fileName)
            check_frame_count = int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            check_cap.release()
            print(f"After second attempt, transformed video contains {check_frame_count} frames")
        
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
         
        if centroids_dom_arr.size == 0 or len(hand_boxes_dom) == 0:
            print("No hand coordinates detected")
            return [], origin, scaling_factor, None

        cap = cv2.VideoCapture(videoDir)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            return [], None, None, None

        last_frame = first_frame.copy()
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 1:
            safe_last_frame_pos = max(0, min(frame_count - 2, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_last_frame_pos)
            ret, potential_last_frame = cap.read()
            if ret:
                last_frame = potential_last_frame
            else:
                print("Could not read near-last frame, using first frame as last frame")
        else:
            print("Video has only one frame, using it as both first and last")
        
        cap.release()

        if len(hand_boxes_dom) == 0:
            print("No hand boxes detected")
            return [], origin, scaling_factor, None

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
            db_dir = "sign_database"
            db_file = "sign_data.json"
            db_data = load_database(db_dir, db_file)
            db_data[fileName] = {
                "name": os.path.splitext(os.path.basename(fileName))[0],
                "features": processed_features,
                "is_one_handed": isOneHanded,
                "duration": float(end_seconds - start_seconds),
                "origin": [float(x) for x in origin] if isinstance(origin, (tuple, list)) else [0.0, 0.0],
                "scaling_factor": float(scaling_factor)
            }
            db_path = os.path.join(db_dir, db_file)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
            with open(db_path, 'w') as f:
                json.dump({"signs": db_data}, f, indent=4)
            print(f"Added sign data to database: {db_path}")

        matches = []
        db_dir = "sign_database"
        db_file = "sign_data.json"
        db_data = load_database(db_dir, db_file)

        if db_data:
            matcher = SignMatcher()
            
            database_signs = []
            sign_names = []
            
            for path, entry in db_data.items():
                if entry['is_one_handed'] != isOneHanded:
                    continue
                database_signs.append(entry['features'])
                sign_names.append(entry['name'])
            
            print(f"Found {len(database_signs)} matching signs in database for comparison")
            
            if database_signs:
                distance_matches = matcher.find_matches(processed_features, database_signs, top_k=10)
                
                for idx, similarity in distance_matches:
                    sign_name = sign_names[idx]
                    matches.append((sign_name, similarity))
                    print(f"Match: {sign_name}, Similarity: {similarity:.2f}%")
            else:
                print("No compatible signs found in database for comparison")

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