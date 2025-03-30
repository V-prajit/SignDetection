import json
import os
import time
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from VideoTrimAndCropping import GetValues
from PyQt6.QtCore import QPoint
import cv2
import numpy as np

class DatabasePopulator:
    def __init__(self, db_dir="sign_database", db_file="sign_data.db", json_backup=True, max_signs=None, 
                 num_workers=None, batch_size=5):
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.db_dir = db_dir
        self.db_file = os.path.join(db_dir, db_file)
        self.json_file = os.path.join(db_dir, "sign_data.json")
        self.benchmark_file = os.path.join(db_dir, "benchmark.json")
        
        self.json_backup = json_backup
        self.max_signs = max_signs
        self.num_workers = num_workers if num_workers else max(1, multiprocessing.cpu_count() - 1)
        self.batch_size = batch_size
        
        self.db_data = self._load_or_create_db()
        self.benchmark_data = {"processing_times": []}
        
        print(f"Initialized with {self.num_workers} worker processes")

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

        return QPoint(start_x, start_y), QPoint(end_x, end_y)

    def _process_video(self, video_entry):
        """Process a single video for parallel execution"""
        parts = [p.strip() for p in video_entry.strip().split(',')]
        video_path = parts[0]
        sign_name = parts[1] if len(parts) > 1 else os.path.splitext(os.path.basename(video_path))[0]
        is_one_handed = True if len(parts) <= 2 else parts[2].lower() == 'true'

        print(f"Processing: {sign_name} from {video_path}")

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            return {"status": "error", "path": video_path, "reason": "not_found"}

        if video_path in self.db_data["signs"]:
            print(f"Video already in database: {video_path}")
            return {"status": "skipped", "path": video_path, "reason": "existing"}

        try:
            video_start_time = time.time()

            duration = self.get_video_duration(video_path)
            if duration is None:
                print(f"Could not read video: {video_path}")
                return {"status": "error", "path": video_path, "reason": "duration_error"}

            start_point, end_point = self.calculate_roi_points(video_path)
            
            matches, origin, scaling_factor, features = GetValues(
                0, 
                duration, 
                start_point,
                end_point,
                video_path,
                is_one_handed
            )

            video_processing_time = time.time() - video_start_time
            
            if features is None or not features:
                print(f"No features extracted for {sign_name}, skipping database entry")
                return {"status": "error", "path": video_path, "reason": "no_features"}
            
            # Return the processed data rather than modifying shared data
            result = {
                "status": "processed",
                "path": video_path,
                "name": sign_name,
                "features": features,
                "is_one_handed": is_one_handed,
                "duration": duration,
                "origin": origin,
                "scaling_factor": scaling_factor,
                "processing_time": video_processing_time
            }
            
            # Clean up temp files
            try:
                base_name = os.path.basename(video_path)
                name_no_ext, ext = os.path.splitext(base_name)
                transformed_video = f"{name_no_ext}_transformed{ext}"
                if os.path.exists(transformed_video):
                    os.remove(transformed_video)
            except Exception as cleanup_error:
                print(f"Error cleaning up transformed video: {str(cleanup_error)}")
                
            return result

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "path": video_path, "reason": str(e)}

    def process_videos(self, video_list_file):
        """Process videos in parallel using multiple processes"""
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
        
        try:
            # Use process pool for parallel execution
            print(f"Processing {total_videos} videos using {self.num_workers} worker processes")
            
            # Process videos in batches
            batch_count = 0
            for i in range(0, len(video_entries), self.batch_size):
                batch = video_entries[i:i+self.batch_size]
                batch_count += 1
                print(f"\nProcessing batch {batch_count} ({len(batch)} videos)")
                
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_video = {executor.submit(self._process_video, entry): entry for entry in batch}
                    
                    for future in as_completed(future_to_video):
                        try:
                            result = future.result()
                            status = result.get("status")
                            
                            if status == "processed":
                                processed_count += 1
                                # Add to database
                                video_path = result["path"]
                                self.db_data["signs"][video_path] = {
                                    "name": result["name"],
                                    "features": result["features"],
                                    "is_one_handed": result["is_one_handed"],
                                    "duration": result["duration"],
                                    "origin": result["origin"],
                                    "scaling_factor": result["scaling_factor"],
                                    "processing_time": result["processing_time"]
                                }
                                
                                self.benchmark_data["processing_times"].append({
                                    "sign_name": result["name"],
                                    "video_path": video_path,
                                    "processing_time_seconds": result["processing_time"],
                                    "video_duration_ms": result["duration"],
                                    "features_count": {
                                        key: len(value) if isinstance(value, list) else "N/A" 
                                        for key, value in result["features"].items()
                                    } if result["features"] else {}
                                })
                                
                                print(f"Successfully processed: {result['name']} (in {result['processing_time']:.2f}s)")
                                
                            elif status == "skipped":
                                skipped_count += 1
                                print(f"Skipped: {result['path']} ({result['reason']})")
                                
                            else:  # error
                                error_count += 1
                                print(f"Error processing: {result['path']} ({result['reason']})")
                                
                        except Exception as e:
                            error_count += 1
                            print(f"Error in future: {str(e)}")
                            traceback.print_exc()
                
                # Save database after each batch
                self._save_db()
                
                # Report progress
                completed = processed_count + skipped_count + error_count
                print(f"Progress: {completed}/{total_videos} ({completed/total_videos*100:.1f}%)")
                print(f"Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
        
        except Exception as e:
            print(f"Error during batch processing: {str(e)}")
            traceback.print_exc()
        
        finally:
            overall_time = time.time() - overall_start_time
            
            self.benchmark_data["summary"] = {
                "total_videos": total_videos,
                "processed_count": processed_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "total_time_seconds": overall_time,
                "average_time_per_video": overall_time / max(processed_count, 1),
                "num_workers": self.num_workers,
                "batch_size": self.batch_size
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
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of videos to process in each batch before saving')
    
    args = parser.parse_args()
    
    populator = DatabasePopulator(
        db_dir=args.db_dir,
        json_backup=not args.no_json,
        max_signs=args.max_signs,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
    
    print(f"Starting database population with settings:")
    print(f"  Max signs: {args.max_signs if args.max_signs else 'All'}")
    print(f"  Database directory: {args.db_dir}")
    print(f"  JSON files: {'Disabled' if args.no_json else 'Enabled'}")
    print(f"  Video list: {args.video_list}")
    print(f"  Worker processes: {populator.num_workers}")
    print(f"  Batch size: {args.batch_size}")
    
    populator.process_videos(args.video_list)

if __name__ == "__main__":
    main()