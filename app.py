
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import sys
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from features import extract_refined_features, extract_all_features, generate_texture_heatmap
from video_analyzer import analyze_video
from analytics import smooth_video_features, detect_change_points, calculate_mes_probability, classify_mes_level, calculate_transition_thresholds

# Reference Baselines (Calibrated from data_sample/ benchmark)
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

# Critical Thresholds for "Top 5" Features (Normal vs. Abnormal cut-off)
REFERENCE_THRESHOLDS = {
    "Contrast": 1.8,      # > threshold is abnormal
    "Homogeneity": 0.75,  # < threshold is abnormal
    "Entropy": 6.30,      # > threshold is abnormal
    "Energy": 0.045,      # < threshold is abnormal
    "ASM": 0.068          # < threshold is abnormal
}

# Centroids for classification (M0-M3)
MAYO_CENTROIDS = {
    "MES 0": {"Contrast": 1.15, "Homogeneity": 0.79, "Entropy": 6.11, "Energy": 0.047, "ASM": 0.073, "Entropy_GLCM": 5.08},
    "MES 1": {"Contrast": 2.45, "Homogeneity": 0.72, "Entropy": 6.18, "Energy": 0.043, "ASM": 0.062, "Entropy_GLCM": 5.51},
    "MES 2": {"Contrast": 2.15, "Homogeneity": 0.74, "Entropy": 6.13, "Energy": 0.043, "ASM": 0.066, "Entropy_GLCM": 5.33},
    "MES 3": {"Contrast": 2.10, "Homogeneity": 0.73, "Entropy": 6.51, "Energy": 0.038, "ASM": 0.055, "Entropy_GLCM": 5.79}
}

def plot_radar_comparison(current_feats):
    """Generates a radar chart comparing current features against M0 and M3 baselines."""
    from math import pi
    
    categories = list(current_feats.keys())
    N = len(categories)
    
    # What's the angle of each axis? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#222831')
    ax.set_facecolor('#393E46')
    
    # Helper to draw a line
    def draw_line(data_dict, label, color, linestyle='solid', alpha=0.25):
        values = [data_dict.get(c, 0) for c in categories]
        # Simple normalization for radar visibility (research used normalized 0-1)
        # Here we just want to show the 'shape'
        norm_v = []
        for i, c in enumerate(categories):
            # Use .get with fallback to avoid KeyErrors if baselines are missing certain features
            b3_val = MAYO_BASELINES["M3 (Severe)"].get(c, 1.0)
            max_v = max(b3_val, data_dict.get(c, 0))
            norm_v.append(data_dict.get(c, 0) / (max_v + 1e-5))
        
        norm_v += norm_v[:1]
        ax.plot(angles, norm_v, linewidth=2, linestyle=linestyle, label=label, color=color)
        ax.fill(angles, norm_v, color=color, alpha=alpha)

    draw_line(MAYO_BASELINES["M0 (Normal)"], "M0 Baseline", "#4E9F3D", linestyle='dashed', alpha=0.1)
    draw_line(MAYO_BASELINES["M3 (Severe)"], "M3 Baseline", "#950101", linestyle='dashed', alpha=0.1)
    draw_line(current_feats, "Current Frame", "#00ADB5", alpha=0.4)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white', size=8)
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#AAAAAA')
    
    plt.title("Texture Signature vs Baselines", color='#00ADB5', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
    
    return fig

# Page Config
st.set_page_config(
    page_title="Colonoscopy Analysis AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .reportview-container {
        background: #222831;
    }
    .sidebar .sidebar-content {
        background: #393E46;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        color: #00ADB5;
    }
    .stButton>button {
        color: #EEEEEE;
        background-color: #00ADB5;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #393E46;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    # st.image("assets/logo.png", width='stretch') # Will enable after generation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="font-size: 3rem;">🧬</span>
        <h2 style="margin: 0; color: #00ADB5;">Colonoscopy AI</h2>
    </div>
    """, unsafe_allow_html=True)
    st.title("Settings")
    
    mode = st.radio("Mode", ["Single Image Analysis", "Video/Real-time"])
    
    st.markdown("---")
    st.markdown("### 🖼️ Processing Params")
    glcm_levels = st.select_slider("GLCM Quantization Levels", options=[8, 16, 32, 64], value=32)
    
    st.markdown("### 🔍 Video Settings")
    analysis_depth = st.radio("Analysis Depth", ["Fast (1 FPS)", "Full (Every Frame)"], index=0)
    show_every_frame = (analysis_depth == "Full (Every Frame)")
    
    # --- New Sliding Window Parameters ---
    st.markdown("### Sliding Window Config")
    win_size = st.slider("Window Size (px)", 16, 128, 32, step=8)
    step_size = st.slider("Step Size (px)", 4, 64, 8, step=4)
    
    st.markdown("---")
    st.markdown("*Developed by Deepmind Agent*")

# Main Area Logic

if mode == "Single Image Analysis":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>Colonoscopy Lesion Analysis</h1>
        <p style="color: #AAAAAA;">Upload a frame to analyze texture anomalies and extract feature vectors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], help="Drop a colonoscopy frame here", label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, channels="BGR", caption="Uploaded Frame", width='stretch')
            
        with col2:
            st.markdown("### Texture Analysis")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            st.markdown("#### Heatmap Overlay")
            heatmap_method = st.selectbox("Method", [
                "None", 
                "Sliding Window (Detailed Mapping)",
                "Local Entropy (Complexity)", 
                "Local Std Dev (Roughness)"
            ])
            
            overlay_img = None
            if heatmap_method != "None":
                alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.4, 0.05)
                
                # --- Advanced Feature Mapping ---
                feat_to_map = "Entropy_GLCM"
                if "Sliding" in heatmap_method:
                    feat_to_map = st.selectbox("Feature to Map", [
                        "Entropy_GLCM", "Correlation", "Contrast", "Homogeneity", "ASM", "MCC"
                    ], help="Select which GLCM feature to project onto the heatmap.")

                method_key = 'sliding_window' if "Sliding" in heatmap_method else ('entropy' if "Entropy" in heatmap_method else 'std')
                
                with st.spinner(f"Generating {heatmap_method}..."):
                    try:
                        texture_map_norm = generate_texture_heatmap(
                            gray, 
                            method=method_key, 
                            win=win_size, 
                            step=step_size, 
                            levels=glcm_levels,
                            feature_to_map=feat_to_map
                        )
                        
                        heatmap_uint8 = (texture_map_norm * 255).astype(np.uint8)
                        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                        
                        if heatmap_color.shape[:2] != image.shape[:2]:
                            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
                            
                        overlay_img = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
                        st.image(overlay_img, channels="BGR", caption=f"{heatmap_method} Overlay", width='stretch')
                        
                        # Display Patch Count (Request: heatmapping count)
                        h, w = texture_map_norm.shape
                        st.metric("Patches Analyzed", f"{h} x {w}", help="Total unique sliding window positions processed.")
                        
                    except Exception as e:
                        st.error(f"Error generating heatmap: {e}")

            st.markdown("---")
            
            with st.spinner("Extracting Refined Features..."):
                try:
                    # UPDATED: Use extract_refined_features
                    feats = extract_refined_features(gray, levels=glcm_levels)
                    feat_df = pd.DataFrame(feats.items(), columns=["Feature", "Value"])
                    st.dataframe(feat_df, hide_index=True)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            st.markdown("### Feature Signature")
            radar_fig = plot_radar_comparison(feats)
            st.pyplot(radar_fig)
            plt.close(radar_fig)

        with col_c2:
            st.markdown("### 🎯 Top 5 Feature Analysis")
            
            # Comparison against Reference Thresholds
            comparison_data = []
            for feat, threshold in REFERENCE_THRESHOLDS.items():
                curr_val = feats.get(feat, 0)
                is_abnormal = False
                if feat in ["Contrast", "Entropy"]:
                    is_abnormal = curr_val > threshold
                else:
                    is_abnormal = curr_val < threshold
                
                status = "🔴 Abnormal" if is_abnormal else "🟢 Normal"
                comparison_data.append({
                    "Feature": feat,
                    "Current": f"{curr_val:.3f}",
                    "Threshold": f"{threshold:.3f}",
                    "Status": status
                })
            
            st.table(pd.DataFrame(comparison_data))

            st.markdown("---")
            st.markdown("### 🧪 Probability & Classification")
            prob = calculate_mes_probability(feats, MAYO_BASELINES)
            predicted_mes = classify_mes_level(feats, MAYO_CENTROIDS)
            
            st.write(f"**Predicted State:** {predicted_mes}")
            st.metric("Inflammation Probability", f"{prob*100:.1f}%", help="Likelihood of active inflammation (MES >= 1) based on multivariate texture distance.")
            
            if prob > 0.7:
                st.error("⚠️ Consistent with Severe Inflammation (likely MES 2-3)")
            elif prob < 0.3:
                st.success("✅ Consistent with Normal/Mild mucosal state (likely MES 0-1)")
            else:
                st.warning(f"ℹ️ Moderate Activity detected ({prob*100:.0f}%)")
            
            st.info("""
            **Reference Note:**  
            The Top 5 features (Contrast, Homogeneity, Entropy, Energy, ASM) are the primary indicators used for threshold-based analysis.
            """)

        st.markdown("---")
        st.markdown("### 📊 Full Feature Distribution Heatmap (20 Features)")
        
        with st.spinner("Calculating full feature set..."):
            all_feats = extract_all_features(gray, levels=glcm_levels)
            all_vals = np.array(list(all_feats.values()))
            all_z = (all_vals - np.mean(all_vals)) / (np.std(all_vals) + 1e-5)
            
            # Reshape for a 4x5 heatmap grid (Original Layout)
            z_grid = all_z.reshape(4, 5)
            feat_names_grid = np.array(list(all_feats.keys())).reshape(4, 5)
            
            fig_z, ax_z = plt.subplots(figsize=(10, 5))
            fig_z.patch.set_facecolor('#222831')
            ax_z.set_facecolor('#222831')
            
            sns.heatmap(z_grid, cmap='viridis', center=0, annot=feat_names_grid, fmt="", cbar=True, ax=ax_z, 
                        yticklabels=False, xticklabels=False, annot_kws={"size": 8, "color": "white"})
            
            # Colorbar styling
            cbar = ax_z.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            st.pyplot(fig_z)
            plt.close(fig_z)
            st.caption("Lower Values (Purple) | Higher Values (Yellow) relative to frame average across all 20 texture metrics.")

elif mode == "Video/Real-time":
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>Video Texture Analysis</h1>
        <p style="color: #AAAAAA;">Analyze colonoscopy video streams for texture anomalies in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        st.markdown("#### Video Heatmap Control")
        heatmap_method_vid = st.selectbox("Heatmap Method (Video)", [
            "None", 
            "Sliding Window (Patch Analysis)",
            "Local Entropy", 
            "Local Std Dev"
        ], key="vid_method")
        
        v_col1, v_col2 = st.columns([2, 1])
        with v_col1:
            video_placeholder = st.empty()
            if heatmap_method_vid != "None" and "Sliding" in heatmap_method_vid:
                feat_to_map_vid = st.selectbox("Feature to Map (Video)", [
                    "Contrast", "Homogeneity", "Entropy", "Energy", "ASM", "Correlation"
                ], key="vid_feat")
            else:
                feat_to_map_vid = "Contrast"

            if st.button("🚀 Run Full Video Analysis (Batch Process)"):
                with st.spinner("Processing every second of video..."):
                    try:
                        # Temp output CSV
                        out_csv = tfile.name + "_features.csv"
                        fps_setting = 0 if show_every_frame else 1.0
                        df_results = analyze_video(tfile.name, output_csv=out_csv, sample_rate_fps=fps_setting, gl_levels=glcm_levels)
                        
                        # Apply Smoothing
                        smooth_win = st.sidebar.slider("Smoothing Window", 1, 11, 5, step=2)
                        df_smooth = smooth_video_features(df_results, window=smooth_win)
                        
                        st.success(f"Analysis complete! Processed {len(df_results)} frames.")
                        
                        # --- NEW: Threshold Calibration UI ---
                        st.markdown("---")
                        st.markdown("### 🎯 Dynamic Threshold Calibration")
                        
                        # Set Default Heuristics
                        current_thresholds = {
                            "Contrast": df_results["Contrast"].mean() * 1.5,
                            "Homogeneity": df_results["Homogeneity"].mean() * 0.8
                        }
                        
                        calc_col1, calc_col2 = st.columns([1, 1])
                        
                        with calc_col1:
                            st.write("**Calibration Source:**")
                            gt_file = st.file_uploader("Upload Ground Truth (CSV)", type=["csv"], help="CSV must have 'frame_index' and 'mes_label'")
                            
                            if gt_file:
                                try:
                                    labels_df = pd.read_csv(gt_file)
                                    current_thresholds = calculate_transition_thresholds(df_results, labels_df)
                                    st.success("Calibration Successful: Thresholds updated from Ground Truth transitions.")
                                except Exception as e:
                                    st.warning(f"Ground Truth format error: {e}. Falling back to default heuristics.")
                            else:
                                st.info("No calibration file uploaded. Currently using Session-Mean heuristics.")

                        with calc_col2:
                            st.write("**Active Thresholds & Baselines:**")
                            thresh_df = pd.DataFrame(current_thresholds.items(), columns=["Feature", "Threshold Value"])
                            st.table(thresh_df)
                            st.caption("Probability Model: Calibrated to M0/M3 Benchmarks")

                        df_final = detect_change_points(df_smooth, current_thresholds, baselines=MAYO_BASELINES, centroids=MAYO_CENTROIDS)
                        
                        # UI Widgets for Analytics
                        st.markdown("### 📈 Advanced Video Analytics")
                        
                        # Display Trend Plot
                        fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
                        fig_trend.patch.set_facecolor('#222831')
                        ax_trend.set_facecolor('#222831')
                        
                        for feat in ["Contrast", "Homogeneity", "Entropy", "MES_Probability"]:
                            color = 'red' if feat == "MES_Probability" else None
                            # Plot Raw (faint)
                            ax_trend.plot(df_results['Timestamp'], df_results.get(feat, df_results['Contrast']), alpha=0.1, color='gray' if not color else color, linestyle='--')
                            # Plot Smoothed
                            lw = 3 if feat == "MES_Probability" else 1.5
                            ax_trend.plot(df_smooth['Timestamp'], df_smooth.get(feat, df_smooth['Contrast']), label=f"{feat}", marker='o', markersize=3, linewidth=lw)
                            
                            # Mark Change Points
                            anomalies = df_final[df_final['Detected_Anomaly']]
                            if not anomalies.empty:
                                ax_trend.scatter(anomalies['Timestamp'], anomalies.get(feat, anomalies['Contrast']), color='red', s=50, edgecolors='white', zorder=5)

                        # Draw Threshold Lines
                        ax_trend.axhline(current_thresholds['Contrast'], color='orange', linestyle=':', label='Contrast Threshold')
                        ax_trend.axhline(0.5, color='white', linestyle='--', alpha=0.3, label='50% Prob Threshold')
                        
                        ax_trend.set_xlabel("Time (s)", color='white')
                        ax_trend.set_ylabel("Value", color='white')
                        ax_trend.tick_params(colors='white')
                        ax_trend.legend(fontsize='x-small', ncol=2)
                        st.pyplot(fig_trend)
                        plt.close(fig_trend)
                        
                        st.info("🔴 Red dots indicate detected texture transitions (potential MES increase).")
                        st.dataframe(df_final, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"Batch Analysis Error: {e}")

        with v_col2:
            st.markdown("#### Real-time Stats")
            fps_placeholder = st.empty()
            mes_placeholder = st.empty()
            feats_placeholder = st.empty()

        # Optimization Parameters
        frame_skip = 1 if show_every_frame else 5 
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            if count % frame_skip != 0:
                continue
                
            start_time = time.time()
            
            # --- Processing ---
            # 1. Downsample for speed
            small_frame = cv2.resize(frame, (400, 400))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            processed_display = frame.copy()
            
            if heatmap_method_vid != "None":
                method_key = 'sliding_window' if "Sliding" in heatmap_method_vid else ('entropy' if "Entropy" in heatmap_method_vid else 'std')
                
                try:
                    from features import generate_texture_heatmap
                    # Use smaller window for video if sliding window to keep it faster
                    v_win = win_size if method_key != 'sliding_window' else max(16, win_size)
                    v_step = step_size if method_key != 'sliding_window' else max(8, step_size)
                    
                    texture_map = generate_texture_heatmap(
                        gray, 
                        method=method_key, 
                        win=v_win, 
                        step=v_step, 
                        levels=glcm_levels,
                        feature_to_map=feat_to_map_vid
                    )
                    
                    heatmap_uint8 = (texture_map * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))
                    
                    processed_display = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                except:
                    pass

            # Update UI
            video_placeholder.image(processed_display, channels="BGR", width='stretch')
            
            end_time = time.time()
            fps = 1.0 / (end_time - start_time + 1e-6)
            fps_placeholder.metric("Processing FPS", f"{fps:.1f}")
            
            # Extract features for current frame
            try:
                current_feats = extract_refined_features(gray, levels=glcm_levels)
                predicted_mes = classify_mes_level(current_feats, MAYO_CENTROIDS)
                
                # Show Predicted MES and Stats
                if "0" not in predicted_mes:
                    mes_placeholder.error(f"**Current State: {predicted_mes}**")
                else:
                    mes_placeholder.success(f"**Current State: {predicted_mes}**")
                
                feat_df_vid = pd.DataFrame(current_feats.items(), columns=["Feature", "Value"])
                feats_placeholder.dataframe(feat_df_vid, hide_index=True)
            except:
                pass
                
            time.sleep(0.01)
            
        cap.release()


