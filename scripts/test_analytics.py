
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from analytics import smooth_video_features, detect_change_points

def test_analytics_logic():
    # 1. Create Mock Data
    data = {
        'Timestamp': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Contrast': [10.0, 11.0, 30.0, 32.0, 31.0], # Jump at 0.3
        'Homogeneity': [0.8, 0.79, 0.4, 0.38, 0.41], # Drop at 0.3
        'Entropy': [5.0, 5.1, 7.0, 7.1, 7.2]
    }
    df = pd.DataFrame(data)
    
    # 2. Test Smoothing
    df_smooth = smooth_video_features(df, window=3)
    print("\nSmoothed Data Sample:")
    print(df_smooth.head())
    
    # 3. Test Change Point Detection
    thresholds = {"Contrast": 20.0, "Homogeneity": 0.5}
    df_final = detect_change_points(df_smooth, thresholds)
    
    anomalies = df_final[df_final['Detected_Anomaly']]
    print(f"\nDetected {len(anomalies)} anomalies.")
    
    if len(anomalies) > 0 and anomalies.iloc[0]['Timestamp'] >= 0.3:
        print("✅ PASSED: Transition detected correctly after smoothing.")
    else:
        print("❌ FAILED: Transition not detected correctly.")

if __name__ == "__main__":
    test_analytics_logic()
