#!/bin/bash
# Extract frames from CLEVRER mp4 videos to jpg for faster data loading
# This significantly speeds up video_prediction training

DATA_ROOT="/cs/data/people/hnam16/data/clevrer_for_savi/videos"

extract_frames() {
    local split=$1
    local split_dir="${DATA_ROOT}/${split}"
    
    echo "Processing ${split} split..."
    
    # Find all mp4 files
    find "${split_dir}" -name "*.mp4" | while read mp4_file; do
        # Get the directory for frames (same name without .mp4)
        frame_dir="${mp4_file%.mp4}"
        
        # Skip if already extracted
        if [ -d "${frame_dir}" ] && [ "$(ls -1 ${frame_dir}/*.jpg 2>/dev/null | wc -l)" -ge 128 ]; then
            continue
        fi
        
        echo "Extracting: ${mp4_file}"
        mkdir -p "${frame_dir}"
        
        # Extract frames as jpg (128 frames per video)
        ffmpeg -i "${mp4_file}" -q:v 2 "${frame_dir}/%06d.jpg" -hide_banner -loglevel error
    done
}

# Extract for both splits
extract_frames "train"
extract_frames "val"

echo "Done! Frames extracted to folders alongside mp4 files."
