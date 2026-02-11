
import cv2
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))
from features import extract_all_features

def main():
    samples = ["M0.jpg", "M1.jpg", "M2.jpg", "M3.jpg"]
    data_dir = "data_sample"
    
    results = []
    for sample in samples:
        path = os.path.join(data_dir, sample)
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = extract_all_features(gray)
        feats["Sample"] = sample
        results.append(feats)
    
    df = pd.DataFrame(results)
    # Print all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
