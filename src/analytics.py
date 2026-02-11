
import pandas as pd
import numpy as np

def smooth_video_features(df, window=5, columns=None):
    """
    Applies a moving average (rolling mean) to smooth noisy texture features.
    
    Args:
        df (pd.DataFrame): DataFrame with frame-by-frame features.
        window (int): Size of the smoothing window.
        columns (list): Columns to smooth. Defaults to Big 5 + Correlation.
        
    Returns:
        pd.DataFrame: DataFrame with smoothed columns.
    """
    if columns is None:
        columns = ["Contrast", "Homogeneity", "Entropy", "Energy", "ASM", "Correlation"]
    
    smoothed_df = df.copy()
    for col in columns:
        if col in df.columns:
            smoothed_df[col] = df[col].rolling(window=window, center=True, min_periods=1).mean()
            
    return smoothed_df

def calculate_transition_thresholds(features_df, labels_df):
    """
    Calculates automated thresholds based on frames where MES transitions occur.
    
    Args:
        features_df (pd.DataFrame): Extracted features (must have 'FrameIndex').
        labels_df (pd.DataFrame): Ground truth labels (must have 'frame_index', 'mes_label').
        
    Returns:
        dict: feature_name -> threshold_value
    """
    # Merge GT labels with extracted features
    merged = pd.merge(features_df, labels_df, left_on='FrameIndex', right_on='frame_index')
    
    # Identify transition frames (where MES changes)
    merged = merged.sort_values('FrameIndex')
    merged['mes_diff'] = merged['mes_label'].diff()
    
    # We focus on the transition from MES 0 to MES 1 (start of inflammation)
    transition_frames = merged[merged['mes_diff'] > 0]
    
    if transition_frames.empty:
        # Fallback to simple mean if no transitions are found in the sampled set
        return merged[["Contrast", "Homogeneity", "Entropy", "Energy", "ASM"]].mean().to_dict()
    
    # Return average feature values at the exact moment of transition
    thresholds = transition_frames[["Contrast", "Homogeneity", "Entropy", "Energy", "ASM"]].mean().to_dict()
    return thresholds

def calculate_mes_probability(feats, baselines):
    """
    Calculates a probability score (0.0 - 1.0) representing the likelihood of 
    active inflammation (MES 1-3) based on distance from known baselines.
    
    Uses weighted contribution from the most reliable monotonic predictors.
    """
    m0 = baselines.get("M0 (Normal)", {})
    m3 = baselines.get("M3 (Severe)", {})
    
    if not m0 or not m3:
        return 0.5 # Default fallback
    
    # Weights for predictors (based on benchmark monotonicity)
    weights = {
        "Entropy": 0.35,
        "Entropy_GLCM": 0.35,
        "Contrast": 0.20,
        "Homogeneity": 0.10
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for feat, weight in weights.items():
        if feat in feats and feat in m0 and feat in m3:
            val = feats[feat]
            low = m0[feat]
            high = m3[feat]
            
            # Normalize feature value to 0-1 based on baseline range
            # Note: Homogeneity decreases, so we invert it
            if feat == "Homogeneity":
                # high (M3) is lower than low (M0)
                norm_val = (low - val) / (low - high + 1e-5)
            else:
                norm_val = (val - low) / (high - low + 1e-5)
            
            # Clamp and accumulate
            norm_val = max(0.0, min(1.0, norm_val))
            total_score += norm_val * weight
            total_weight += weight
            
    if total_weight == 0:
        return 0.0
        
    return total_score / total_weight

def classify_mes_level(feats, centroids):
    """
    Finds the closest MES level (0, 1, 2, or 3) using weighted Euclidean distance.
    Returns a string like 'MES 0', 'MES 1', etc.
    """
    weights = {
        "Entropy": 1.0,
        "Entropy_GLCM": 1.2,
        "Contrast": 0.8,
        "Homogeneity": 0.5
    }
    
    min_dist = float('inf')
    closest_label = "N/A"
    
    for label, center in centroids.items():
        dist = 0.0
        for feat, w in weights.items():
            if feat in feats and feat in center:
                dist += w * (feats[feat] - center[feat])**2
        
        dist = np.sqrt(dist)
        if dist < min_dist:
            min_dist = dist
            closest_label = label
            
    return closest_label

def detect_change_points(df, thresholds, baselines=None, centroids=None):
    """
    Detects points in time where features cross calculated thresholds.
    Also calculates frame-by-frame probability and classification.
    
    Args:
        df (pd.DataFrame): Smoothed feature DataFrame.
        thresholds (dict): Feature thresholds.
        baselines (dict): Optional M0/M3 baselines for probability scoring.
        centroids (dict): Optional M0-M3 centroids for specific classification.
        
    Returns:
        pd.DataFrame: Original DF with 'Detected_Anomaly', 'MES_Probability', and 'Predicted_MES'.
    """
    df = df.copy()
    df['Anomaly_Score'] = 0
    
    # Simple voting mechanism
    if 'Contrast' in thresholds and 'Contrast' in df.columns:
        df['Anomaly_Score'] += (df['Contrast'] > thresholds['Contrast']).astype(int)
    
    if 'Homogeneity' in thresholds and 'Homogeneity' in df.columns:
        df['Anomaly_Score'] += (df['Homogeneity'] < thresholds['Homogeneity']).astype(int)
        
    df['Detected_Anomaly'] = df['Anomaly_Score'] >= 1
    
    # Probability Tracking
    if baselines:
        df['MES_Probability'] = df.apply(lambda row: calculate_mes_probability(row.to_dict(), baselines), axis=1)
    else:
        df['MES_Probability'] = df['Anomaly_Score'] / 2.0

    # Specific MES Classification
    if centroids:
        df['Predicted_MES'] = df.apply(lambda row: classify_mes_level(row.to_dict(), centroids), axis=1)
    else:
        df['Predicted_MES'] = df['MES_Probability'].apply(lambda p: f"MES {round(p*3)}")
    
    return df
