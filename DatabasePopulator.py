import json
import os
from VideoTrimAndCropping import GetValues
from PyQt6.QtCore import QPoint
import cv2
import numpy as np

class DatabasePopulator:
    def __init__(self, db_file="sign_database.json"):
        self.db_file = db_file
        self.db_data = self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {"signs": {}}

    def _save_db(self):
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

        with open(self.db_file, 'w') as f:
            json.dump({"signs": json_safe_data}, f, indent=4)

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

    def process_videos(self, video_list_file):
        if not os.path.exists(video_list_file):
            print(f"Error: Video list file not found: {video_list_file}")
            return

        with open(video_list_file, 'r') as f:
            video_entries = f.readlines()

        for entry in video_entries:
            parts = [p.strip() for p in entry.strip().split(',')]
            video_path = parts[0]
            
            sign_name = parts[1] if len(parts) > 1 else os.path.splitext(os.path.basename(video_path))[0]
            is_one_handed = True if len(parts) <= 2 else parts[2].lower() == 'true'

            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue

            if video_path in self.db_data["signs"]:
                print(f"Video already in database: {video_path}")
                continue

            print(f"Processing {video_path}...")

            duration = self.get_video_duration(video_path)
            if duration is None:
                print(f"Could not read video: {video_path}")
                continue

            start_point, end_point = self.calculate_roi_points(video_path)
            
            try:
                matches, origin, scaling_factor, features = GetValues(
                    0, 
                    duration, 
                    start_point,
                    end_point,
                    video_path,
                    is_one_handed
                )

                self.db_data["signs"][video_path] = {
                    "name": sign_name,
                    "features": features,
                    "is_one_handed": is_one_handed,
                    "duration": duration,
                    "origin": origin,
                    "scaling_factor": scaling_factor
                }

                print(f"Successfully added {sign_name} to database")
                self._save_db()

                base_name = os.path.basename(video_path)
                name_no_ext, ext = os.path.splitext(base_name)
                transformed_video = f"{name_no_ext}_transformed{ext}"
                if os.path.exists(transformed_video):
                    os.remove(transformed_video)
                    print(f"Deleted transformed video: {transformed_video}")

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                try:
                    base_name = os.path.basename(video_path)
                    name_no_ext, ext = os.path.splitext(base_name)
                    transformed_video = f"{name_no_ext}_transformed{ext}"
                    if os.path.exists(transformed_video):
                        os.remove(transformed_video)
                        print(f"Deleted transformed video: {transformed_video}")
                except Exception as cleanup_error:
                    print(f"Error cleaning up transformed video: {str(cleanup_error)}")

def main():
    populator = DatabasePopulator()
    video_list_file = "videos_to_add.txt"
    populator.process_videos(video_list_file)

if __name__ == "__main__":
    main()