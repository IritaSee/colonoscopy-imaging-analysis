import time
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import extract_refined_features

def test():
    print("Testing extract_refined_features execution time...")
    # Create a dummy grayscale image 400x400 as used in video_analyzer.py
    gray_img = np.random.randint(0, 255, (400, 400), dtype=np.uint8)
    
    # Warmup
    _ = extract_refined_features(gray_img, levels=32)
    
    # Measure
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        _ = extract_refined_features(gray_img, levels=32)
    end = time.time()
    
    avg_time = (end - start) / n_iters
    print(f"Average time per frame (400x400): {avg_time:.4f} seconds")
    print(f"Frames per second (FPS): {1/avg_time:.2f}")
    
    # Example video calc
    # A 1-minute video at 1 FPS sampling
    print(f"\nEstimated time for 1-minute video (sampled 1 frame/sec, i.e., 60 frames): {avg_time * 60:.2f} seconds")
    print(f"Estimated time for 1-minute video (sampled 30 frames/sec, i.e., 1800 frames): {avg_time * 1800:.2f} seconds ({avg_time * 1800 / 60:.2f} minutes)")

if __name__ == "__main__":
    test()
