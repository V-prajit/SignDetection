import json
import os
from typing import Dict, List, Optional
import cv2
from datetime import datetime

class VideoManager:
    def __init__(self, json_path: str = "video_database.json"):
        self.json_path = json_path
        self.video_data = self._load_json()

    def _load_json(self) -> Dict:
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return {
            "videos": {},
            "last_updated": str(datetime.now())
        }

    def _save_json(self):
        self.video_data["last_updated"] = str(datetime.now())
        with open(self.json_path, 'w') as f:
            json.dump(self.video_data, f, indent=4)

    def get_video_duration(self, video_path: str) -> Optional[float]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (frame_count / fps) * 1000
            
            cap.release()
            return duration
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return None

    def add_video(self, video_path: str, sign_info: Dict, roi_start: tuple = (100, 100), 
                 roi_end: tuple = (400, 400), is_one_handed: bool = True) -> bool:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
        duration = self.get_video_duration(video_path)
        if duration is None:
            print(f"Error: Could not read video file: {video_path}")
            return False

        video_entry = {
            "path": video_path,
            "added_date": str(datetime.now()),
            "duration": duration,
            "sign_info": sign_info,
            "processing_info": {
                "roi_start": roi_start,
                "roi_end": roi_end,
                "is_one_handed": is_one_handed,
                "start_time": 0,
                "end_time": duration
            },
            "processed": False
        }

        video_id = os.path.basename(video_path)
        self.video_data["videos"][video_id] = video_entry
        self._save_json()
        
        print(f"Successfully added video: {video_path}")
        return True

    def get_unprocessed_videos(self) -> List[Dict]:
        return [
            video for video in self.video_data["videos"].values()
            if not video["processed"]
        ]

    def mark_video_processed(self, video_path: str):
        video_id = os.path.basename(video_path)
        if video_id in self.video_data["videos"]:
            self.video_data["videos"][video_id]["processed"] = True
            self._save_json()

    def is_video_in_database(self, video_path: str) -> bool:
        video_id = os.path.basename(video_path)
        return video_id in self.video_data["videos"]