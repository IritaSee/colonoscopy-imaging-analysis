
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from features import generate_texture_heatmap

def test_sliding_window():
    print("Testing Sliding Window Heatmap...")
    # Create a dummy image (H, W)
    img = (np.random.rand(128, 128) * 255).astype(np.uint8)
    
    # Test parameters
    win = 32
    step = 8
    
    try:
        heatmap = generate_texture_heatmap(img, method='sliding_window', win=win, step=step)
        
        # Verify shape (should match original image because of cv2.resize in the function)
        assert heatmap.shape == img.shape, f"Expected shape {img.shape}, got {heatmap.shape}"
        
        # Verify range
        assert np.min(heatmap) >= 0 and np.max(heatmap) <= 1, "Heatmap values out of [0, 1] range"
        
        print("✅ Sliding Window Verification Successful!")
        print(f"Heatmap Shape: {heatmap.shape}")
        print(f"Value Range: [{np.min(heatmap):.2f}, {np.max(heatmap):.2f}]")
        
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_sliding_window()
