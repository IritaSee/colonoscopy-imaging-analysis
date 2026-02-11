
import cv2
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from features import extract_refined_features

def test_feature_progression():
    samples = {
        "M0": "data_sample/M0.jpg",
        "M1": "data_sample/M1.jpg",
        "M2": "data_sample/M2.jpg",
        "M3": "data_sample/M3.jpg"
    }
    
    results = []
    for key, path in samples.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not load {path}")
            continue
            
        feats = extract_refined_features(img)
        feats['Sample'] = key
        results.append(feats)
        
    df = pd.DataFrame(results).set_index('Sample')
    print("\nTexture Feature Progression Check (M0 vs M3):")
    print(df[['Contrast', 'Homogeneity', 'Entropy', 'Energy', 'ASM', 'Correlation']])
    
    # Simple validation logic
    if df.loc['M3', 'Contrast'] > df.loc['M0', 'Contrast']:
        print("\n✅ PASSED: Contrast increases with severity.")
    else:
        print("\n❌ FAILED: Contrast did not increase with severity.")
        
    if df.loc['M3', 'Homogeneity'] < df.loc['M0', 'Homogeneity']:
        print("✅ PASSED: Homogeneity decreases with severity.")
    else:
        print("❌ FAILED: Homogeneity did not decrease with severity.")

if __name__ == "__main__":
    test_frame = 'data_sample/M0.jpg'
    if os.path.exists(test_frame):
        test_feature_progression()
    else:
        print(f"Missing sample data: {test_frame}")
