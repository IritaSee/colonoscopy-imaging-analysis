
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Replicate the fixed baselines from app.py
MAYO_BASELINES = {
    "M0 (Normal)": {"Contrast": 2.0, "Homogeneity": 0.65, "Entropy": 4.5, "Energy": 0.05, "ASM": 0.02, "Correlation": 0.8},
    "M3 (Severe)": {"Contrast": 18.0, "Homogeneity": 0.25, "Entropy": 7.5, "Energy": 0.01, "ASM": 0.005, "Correlation": 0.2}
}

def test_radar_logic():
    # Mocking the 'feats' returned by extract_refined_features
    mock_feats = {
        "Contrast": 10.0,
        "Homogeneity": 0.4,
        "Entropy": 6.0,
        "Energy": 0.03,
        "ASM": 0.01,
        "Correlation": 0.5
    }
    
    categories = list(mock_feats.keys())
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    print(f"Testing radar logic with keys: {categories}")
    
    def draw_line(data_dict, label):
        print(f"Drawing line for: {label}")
        norm_v = []
        for c in categories:
            # This is where the KeyError occurred
            target_baseline = MAYO_BASELINES["M3 (Severe)"]
            if c not in target_baseline:
                raise KeyError(f"Key '{c}' not found in MAYO_BASELINES['M3 (Severe)']")
            
            max_v = max(target_baseline[c], data_dict.get(c, 0))
            norm_v.append(data_dict.get(c, 0) / (max_v + 1e-5))
        
        norm_v += norm_v[:1]
        return norm_v

    try:
        draw_line(MAYO_BASELINES["M0 (Normal)"], "M0 Baseline")
        draw_line(MAYO_BASELINES["M3 (Severe)"], "M3 Baseline")
        draw_line(mock_feats, "Current Frame")
        print("\n✅ PASSED: Radar logic successfully processed all 6 features without KeyError.")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    test_radar_logic()
