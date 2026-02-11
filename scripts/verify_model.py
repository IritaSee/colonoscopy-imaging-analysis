
import cv2
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))
from features import extract_refined_features
from analytics import calculate_mes_probability

MAYO_BASELINES = {
    "M0 (Normal)": {
        "Contrast": 1.15, "Homogeneity": 0.79, "Entropy": 6.11, 
        "Energy": 0.047, "ASM": 0.073, "Correlation": 0.97, 
        "Entropy_GLCM": 5.08
    },
    "M3 (Severe)": {
        "Contrast": 2.10, "Homogeneity": 0.73, "Entropy": 6.51, 
        "Energy": 0.038, "ASM": 0.055, "Correlation": 0.96, 
        "Entropy_GLCM": 5.79
    }
}

def main():
    samples = ["M0.jpg", "M1.jpg", "M2.jpg", "M3.jpg"]
    data_dir = "data_sample"
    
    print("Verification of Probability Model:")
    print("-" * 40)
    for sample in samples:
        path = os.path.join(data_dir, sample)
        img = cv2.imread(path)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = extract_refined_features(gray)
        prob = calculate_mes_probability(feats, MAYO_BASELINES)
        
        print(f"Sample: {sample}")
        print(f"  - Entropy: {feats['Entropy']:.3f}")
        print(f"  - Entropy_GLCM: {feats['Entropy_GLCM']:.3f}")
        print(f"  - Contrast: {feats['Contrast']:.3f}")
        print(f"  - Homogeneity: {feats['Homogeneity']:.3f}")
        print(f"  => Probability: {prob*100:.1f}%")
        print("-" * 40)

if __name__ == "__main__":
    main()
