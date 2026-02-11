
import cv2
import numpy as np
import pandas as pd
import os
import time
from features import extract_refined_features

def analyze_video(video_path, output_csv=None, sample_rate_fps=1.0, resize_dims=(400, 400), gl_levels=32):
    """
    Analyzes a video file frame-by-frame (sampled) and extracts refined texture features.
    
    Args:
        video_path (str): Path to the input video file.
        output_csv (str): Path to save the extracted feature CSV.
        sample_rate_fps (float): Number of frames to process per second of video.
        resize_dims (tuple): Dimensions to resize frames for faster processing.
        
    Returns:
        pd.DataFrame: DataFrame containing per-frame features.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    
    if sample_rate_fps <= 0:
        frame_interval = 1
    else:
        frame_interval = int(video_fps / sample_rate_fps) if sample_rate_fps > 0 else 1
    
    frame_interval = max(1, frame_interval)
    
    print(f"Analyzing Video: {os.path.basename(video_path)}")
    print(f"FPS: {video_fps:.2f} | Total Frames: {total_frames} | Duration: {duration_sec:.1f}s")
    print(f"Sampling every {frame_interval} frames (~{sample_rate_fps} FPS)")
    
    all_frame_data = []
    frame_idx = 0
    processed_count = 0
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            # Timestamp in video
            timestamp = frame_idx / video_fps
            
            # Preprocess
            small_frame = cv2.resize(frame, resize_dims)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Extract features
            try:
                feats = extract_refined_features(gray, levels=gl_levels)
                feats['Timestamp'] = round(timestamp, 2)
                feats['FrameIndex'] = frame_idx
                all_frame_data.append(feats)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed_count} frames... (Elapsed: {elapsed:.1f}s)")
            except Exception as e:
                print(f"Error at frame {frame_idx}: {e}")
                
        frame_idx += 1
        
    cap.release()
    
    df = pd.DataFrame(all_frame_data)
    
    # Reorder columns to put metadata first
    cols = ['FrameIndex', 'Timestamp'] + [c for c in df.columns if c not in ['FrameIndex', 'Timestamp']]
    df = df[cols]
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Analysis complete. Results saved to: {output_csv}")
        
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract texture features from video frames.")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate (default: 1.0 FPS)")
    parser.add_argument("--out", help="Output CSV path")
    
    args = parser.parse_args()
    
    output = args.out or args.video.rsplit('.', 1)[0] + "_texture_features.csv"
    analyze_video(args.video, output_csv=output, sample_rate_fps=args.fps)
