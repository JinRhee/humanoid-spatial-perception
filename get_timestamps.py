import os
import csv
from pathlib import Path

def get_timestamps_from_directory(image_dir, output_csv):
    """
    Read image files and create a CSV with filenames and timestamps in nanoseconds.
    
    Image format: image_{sec:010d}_{nsec:09d}.jpg
    """
    image_files = sorted(Path(image_dir).glob("image_*.jpg"))
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'timestamp_ns'])
        
        for image_file in image_files:
            # Parse filename: image_0000000001_000000000.jpg
            stem = image_file.stem  # Remove .jpg
            parts = stem.split('_')
            
            if len(parts) == 3:
                sec = int(parts[1])
                nsec = int(parts[2])
                timestamp_ns = sec * 1_000_000_000 + nsec
                
                writer.writerow([image_file.name, timestamp_ns])

if __name__ == "__main__":
    image_directory = "/home/ori/data/AprilTag1_ros_sequence_6Hz/images/cam1"
    output_file = "/home/ori/data/AprilTag1_ros_sequence_6Hz/images/short_times.csv"
    get_timestamps_from_directory(image_directory, output_file)
    print(f"CSV file created: {output_file}")