import json
import os
import time
from VideoTrimAndCropping import GetValues
from PyQt6.QtCore import QPoint
import cv2
import numpy as np

class DatabasePopulator:
    def __init__(self, db_dir="sign_database", db_file="sign_data.db", json_backup=True, max_signs=None):
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.db_dir = db_dir
        self.db_file = os.path.join(db_dir, db_file)
        self.json_file = os.path.join(db_dir, "sign_data.json")
        self.benchmark_file = os.path.join(db_dir, "benchmark.json")
        
        self.json_backup = json_backup
        self.max_signs = max_signs
        self.db_data = self._load_or_create_db()
        self.benchmark_data = {"processing_times": []}

    def _load_or_create_db(self):
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                return json.load(f)
        return {"signs": {}}

    def _save_db(self):
        try:
            print("\n--- DATABASE SAVE ATTEMPT ---")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Database directory: {os.path.abspath(self.db_dir)}")
            
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return [[None if np.isnan(x) else float(x) for x in row] 
                        for row in obj]
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj

            json_safe_data = {}
            for video_path, video_data in self.db_data["signs"].items():
                json_safe_data[video_path] = {
                    "name": video_data["name"],
                    "features": {
                        key: convert_to_json_serializable(value)
                        for key, value in video_data["features"].items()
                    },
                    "is_one_handed": video_data["is_one_handed"],
                    "duration": float(video_data["duration"]),
                    "origin": convert_to_json_serializable(video_data["origin"]),
                    "scaling_factor": float(video_data["scaling_factor"])
                }
                if "processing_time" in video_data:
                    json_safe_data[video_path]["processing_time"] = float(video_data["processing_time"])

            print(f"Number of signs in database: {len(json_safe_data)}")
            
            print(f"Saving JSON data to: {self.json_file}")
            with open(self.json_file, 'w') as f:
                json.dump({"signs": json_safe_data}, f, indent=4)
            
            print(f"Saving DB data to: {self.db_file}")
            with open(self.db_file, 'w') as f:
                json.dump({"signs": json_safe_data}, f, indent=4)
                
            if self.benchmark_data["processing_times"]:
                print(f"Saving benchmark data to: {self.benchmark_file}")
                with open(self.benchmark_file, 'w') as f:
                    json.dump(self.benchmark_data, f, indent=4)
                
            files_exist = os.path.exists(self.json_file) and os.path.exists(self.db_file)
            if files_exist:
                json_size = os.path.getsize(self.json_file)
                db_size = os.path.getsize(self.db_file)
                print(f"Database successfully saved!")
                print(f"  JSON file size: {json_size} bytes")
                print(f"  DB file size: {db_size} bytes")
            else:
                print(f"ERROR: One or more files do not exist after saving attempt")
        
        except Exception as e:
            print(f"ERROR DURING DATABASE SAVE: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) * 1000
        
        cap.release()
        return duration

    def calculate_roi_points(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}, using default ROI")
            return QPoint(100, 100), QPoint(400, 400)
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        padding_percent = 0.20
        start_x = int(width * padding_percent)
        start_y = int(height * padding_percent)
        end_x = int(width * (1 - padding_percent))
        end_y = int(height * (1 - padding_percent))

        MIN_SIZE = 200
        if (end_x - start_x) < MIN_SIZE:
            center_x = (start_x + end_x) // 2
            start_x = center_x - MIN_SIZE // 2
            end_x = center_x + MIN_SIZE // 2
        
        if (end_y - start_y) < MIN_SIZE:
            center_y = (start_y + end_y) // 2
            start_y = center_y - MIN_SIZE // 2
            end_y = center_y + MIN_SIZE // 2

        start_x = max(0, min(start_x, width - MIN_SIZE))
        start_y = max(0, min(start_y, height - MIN_SIZE))
        end_x = max(MIN_SIZE, min(end_x, width))
        end_y = max(MIN_SIZE, min(end_y, height))

        print(f"Calculated ROI: ({start_x}, {start_y}) to ({end_x}, {end_y})")
        return QPoint(start_x, start_y), QPoint(end_x, end_y)

    # Modified to extract feature extraction logic from GetValues
    def extract_features_only(self, video_path, is_one_handed):
        """Extract features from a video without comparing to database"""
        from faceDetection import detect_face
        from HandCoordinates import HandCoordinates
        from hand_processing import extract_hand_image, preprocess_hand_image
        from LinearInterpolation import InterpolateAndResample
        import os
        import cv2
        
        print(f"Extracting features from: {video_path}")
        
        # Get video duration
        duration = self.get_video_duration(video_path)
        if duration is None:
            print(f"Could not read video: {video_path}")
            return None, None, None, None
            
        # Calculate ROI
        start_point, end_point = self.calculate_roi_points(video_path)
        
        # Detect face for normalization
        origin, scaling_factor, _ = detect_face(video_path)
        if origin is None or scaling_factor is None:
            print("Face detection failed, using default normalization parameters")
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            origin = (width/2, height/2)
            scaling_factor = 1.0/height
            
        print(f"Using normalization - Origin: {origin}, Scaling: {scaling_factor}")
        
        # Extract hand coordinates
        (centroids_dom_arr,
         centroids_nondom_arr,
         hand_boxes_dom,
         hand_boxes_nondom,
         _,  # origin is already set
         l_delta_arr,
         orientation_dom_arr,
         orientation_nondom_arr,
         orientation_delta_arr) = HandCoordinates(video_path, origin, scaling_factor, is_one_handed)
        
        if centroids_dom_arr.size == 0 or len(hand_boxes_dom) == 0:
            print("No hand coordinates detected")
            return None, None, None, None
            
        # Extract hand images from first and last frame
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read first frame")
            return None, None, None, None
            
        last_frame = first_frame.copy()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 1:
            safe_last_frame_pos = max(0, min(frame_count - 2, frame_count - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_last_frame_pos)
            ret, potential_last_frame = cap.read()
            if ret:
                last_frame = potential_last_frame
        cap.release()
        
        # Process hand images
        dom_start_img = extract_hand_image(first_frame, hand_boxes_dom[0])
        dom_end_img = extract_hand_image(last_frame, hand_boxes_dom[-1])
        
        H_d_s = preprocess_hand_image(dom_start_img)
        H_d_e = preprocess_hand_image(dom_end_img)
        
        # Interpolate to standardize feature length
        TARGET_FRAMES = 20
        Interpolated_Dominant_Hand = InterpolateAndResample(centroids_dom_arr, TARGET_FRAMES)
        Interpolated_nonDominant_Hand = InterpolateAndResample(centroids_nondom_arr, TARGET_FRAMES)
        Interpolated_l_Delta = InterpolateAndResample(l_delta_arr, TARGET_FRAMES)
        Interpolated_orient_dom = InterpolateAndResample(orientation_dom_arr, TARGET_FRAMES)
        Interpolated_orient_nondom = InterpolateAndResample(orientation_nondom_arr, TARGET_FRAMES)
        Interpolated_orient_delta = InterpolateAndResample(orientation_delta_arr, TARGET_FRAMES)
        
        # Create features dictionary
        processed_features = {
            'centroids_dom_arr': Interpolated_Dominant_Hand.tolist(),
            'centroids_nondom_arr': Interpolated_nonDominant_Hand.tolist(),
            'l_delta_arr': Interpolated_l_Delta.tolist(),
            'orientation_dom_arr': Interpolated_orient_dom.tolist(),
            'orientation_nondom_arr': Interpolated_orient_nondom.tolist(),
            'orientation_delta_arr': Interpolated_orient_delta.tolist(),
            'H_d_s': H_d_s,
            'H_d_e': H_d_e,
            'is_one_handed': is_one_handed,
            'frame_count': TARGET_FRAMES
        }
        
        # Add non-dominant hand features if applicable
        if not is_one_handed and len(hand_boxes_nondom) > 0:
            nondom_start_img = extract_hand_image(first_frame, hand_boxes_nondom[0])
            nondom_end_img = extract_hand_image(last_frame, hand_boxes_nondom[-1])
            
            H_nd_s = preprocess_hand_image(nondom_start_img)
            H_nd_e = preprocess_hand_image(nondom_end_img)
            
            processed_features.update({
                'H_nd_s': H_nd_s,
                'H_nd_e': H_nd_e
            })
            
        return processed_features, origin, scaling_factor, duration

    def process_videos(self, video_list_file):
        if not os.path.exists(video_list_file):
            print(f"Error: Video list file not found: {video_list_file}")
            return

        with open(video_list_file, 'r') as f:
            video_entries = f.readlines()
            
        if self.max_signs is not None:
            print(f"Limiting processing to {self.max_signs} signs as requested")
            video_entries = video_entries[:self.max_signs]
        
        total_videos = len(video_entries)
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        overall_start_time = time.time()

        for i, entry in enumerate(video_entries, 1):
            parts = [p.strip() for p in entry.strip().split(',')]
            video_path = parts[0]
            
            sign_name = parts[1] if len(parts) > 1 else os.path.splitext(os.path.basename(video_path))[0]
            is_one_handed = True if len(parts) <= 2 else parts[2].lower() == 'true'

            print(f"[{i}/{total_videos}] Processing: {sign_name}")

            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                error_count += 1
                continue

            if video_path in self.db_data["signs"]:
                print(f"Video already in database: {video_path}")
                skipped_count += 1
                continue

            print(f"Processing {video_path}...")

            video_start_time = time.time()

            try:
                # Use our new method to extract features directly without comparing
                features, origin, scaling_factor, duration = self.extract_features_only(
                    video_path, 
                    is_one_handed
                )

                video_processing_time = time.time() - video_start_time
                
                if features is None or not features:
                    print(f"No features extracted for {sign_name}, skipping database entry")
                    error_count += 1
                    continue
                
                self.db_data["signs"][video_path] = {
                    "name": sign_name,
                    "features": features,
                    "is_one_handed": is_one_handed,
                    "duration": duration,
                    "origin": origin,
                    "scaling_factor": scaling_factor,
                    "processing_time": video_processing_time
                }
                
                self.benchmark_data["processing_times"].append({
                    "sign_name": sign_name,
                    "video_path": video_path,
                    "processing_time_seconds": video_processing_time,
                    "video_duration_ms": duration,
                    "features_count": {
                        key: len(value) if isinstance(value, list) else "N/A" 
                        for key, value in features.items()
                    } if features else {}
                })

                print(f"Successfully added {sign_name} to database (processed in {video_processing_time:.2f}s)")
                processed_count += 1
                
                if processed_count > 0 and processed_count % 10 == 0:
                    self._save_db()

                # Clean up any temporary files
                base_name = os.path.basename(video_path)
                name_no_ext, ext = os.path.splitext(base_name)
                transformed_video = f"{name_no_ext}_transformed{ext}"
                if os.path.exists(transformed_video):
                    os.remove(transformed_video)
                    print(f"Deleted transformed video: {transformed_video}")

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                error_count += 1
                try:
                    base_name = os.path.basename(video_path)
                    name_no_ext, ext = os.path.splitext(base_name)
                    transformed_video = f"{name_no_ext}_transformed{ext}"
                    if os.path.exists(transformed_video):
                        os.remove(transformed_video)
                        print(f"Deleted transformed video: {transformed_video}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up transformed video: {str(cleanup_error)}")
        
        overall_time = time.time() - overall_start_time
        
        self.benchmark_data["summary"] = {
            "total_videos": total_videos,
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "total_time_seconds": overall_time,
            "average_time_per_video": overall_time / max(processed_count, 1)
        }
        
        print("\n--- PROCESSING SUMMARY ---")
        print(f"Total videos in list: {total_videos}")
        print(f"Videos processed: {processed_count}")
        print(f"Videos skipped: {skipped_count}")
        print(f"Errors: {error_count}")
        print(f"Total processing time: {overall_time:.2f}s")
        print(f"Average time per video: {overall_time / max(processed_count, 1):.2f}s")
        
        self._save_db()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sign language videos for database.')
    parser.add_argument('--max_signs', type=int, help='Maximum number of signs to process')
    parser.add_argument('--db_dir', type=str, default="sign_database",
                      help='Directory to store database files')
    parser.add_argument('--video_list', type=str, default="videos_to_add.txt",
                      help='Path to video list file')
    parser.add_argument('--no_json', action='store_true', 
                      help='Disable JSON file creation (for release)')
    
    args = parser.parse_args()
    
    populator = DatabasePopulator(
        db_dir=args.db_dir,
        json_backup=not args.no_json,
        max_signs=args.max_signs
    )
    
    print(f"Starting database population with settings:")
    print(f"  Max signs: {args.max_signs if args.max_signs else 'All'}")
    print(f"  Database directory: {args.db_dir}")
    print(f"  JSON files: {'Disabled' if args.no_json else 'Enabled'}")
    print(f"  Video list: {args.video_list}")
    
    populator.process_videos(args.video_list)

if __name__ == "__main__":
    main()