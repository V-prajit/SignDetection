import os
import argparse

def generate_videos_list(root_dir, output_file="videos_to_add.txt"):
    print(f"Scanning directory: {root_dir}")
    
    video_files = []
    video_extensions = ['.mpg', '.mpeg', '.mp4']
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                full_path = os.path.abspath(os.path.join(dirpath, filename))

                sign_name = os.path.splitext(filename)[0]
                
                is_one_handed = "true"
                video_files.append(f"{full_path},{sign_name},{is_one_handed}")

    with open(output_file, 'w') as f:
        for video_file in video_files:
            f.write(f"{video_file}\n")
    
    print(f"Found {len(video_files)} video files. List written to {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a list of video files for SignDetection")
    parser.add_argument("root_dir", help="Root directory to scan for video files")
    parser.add_argument("--output", "-o", default="videos_to_add.txt", 
                        help="Output file path (default: videos_to_add.txt)")
    args = parser.parse_args()
    
    generate_videos_list(args.root_dir, args.output)