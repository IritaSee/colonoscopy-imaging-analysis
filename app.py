
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

from features import extract_all_features

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
    # st.image("assets/logo.png", use_container_width=True) # Will enable after generation
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="font-size: 3rem;">🧬</span>
        <h2 style="margin: 0; color: #00ADB5;">Colonoscopy AI</h2>
    </div>
    """, unsafe_allow_html=True)
    st.title("Settings")
    
    mode = st.radio("Mode", ["Single Image Analysis", "Video/Real-time"])
    
    st.markdown("### Analysis Parameters")
    glcm_levels = st.slider("GLCM Levels", 8, 256, 32, step=8)
    
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
            st.image(image, channels="BGR", caption="Uploaded Frame", use_container_width=True)
            
        with col2:
            st.markdown("### Texture Analysis")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            st.markdown("#### Heatmap Overlay")
            heatmap_method = st.selectbox("Method", [
                "None", 
                "Sliding Window (Detailed Patch Analysis)",
                "Local Entropy (Complexity)", 
                "Local Std Dev (Roughness)"
            ])
            
            overlay_img = None
            if heatmap_method != "None":
                alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.4, 0.05)
                
                method_key = 'sliding_window' if "Sliding" in heatmap_method else ('entropy' if "Entropy" in heatmap_method else 'std')
                
                with st.spinner(f"Generating {heatmap_method}..."):
                    try:
                        from features import generate_texture_heatmap
                        texture_map_norm = generate_texture_heatmap(
                            gray, 
                            method=method_key, 
                            win=win_size, 
                            step=step_size, 
                            levels=glcm_levels
                        )
                        
                        heatmap_uint8 = (texture_map_norm * 255).astype(np.uint8)
                        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                        
                        if heatmap_color.shape[:2] != image.shape[:2]:
                            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
                            
                        overlay_img = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
                        st.image(overlay_img, channels="BGR", caption=f"{heatmap_method} Overlay", use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating heatmap: {e}")

            st.markdown("---")
            
            with st.spinner("Extracting Features..."):
                try:
                    feats = extract_all_features(gray, levels=glcm_levels)
                    feat_df = pd.DataFrame(feats.items(), columns=["Feature", "Value"])
                    st.dataframe(feat_df, height=200)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("### Feature Distribution Heatmap")
        
        current_vector = np.array(list(feats.values()))
        z_scores = (current_vector - np.mean(current_vector)) / (np.std(current_vector) + 1e-5)
        
        # Reshape according to actual feature count (6 FO + 14 GLCM = 20)
        z_grid = z_scores.reshape(4, 5)
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#222831')
        ax.set_facecolor('#222831')
        
        sns.heatmap(z_grid, cmap='viridis', center=0, annot=True, fmt=".1f", cbar=True, ax=ax, 
                    yticklabels=False, xticklabels=False)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        st.pyplot(fig)
        plt.close(fig)

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
        with v_col2:
            st.markdown("#### Stats")
            fps_placeholder = st.empty()
            feats_placeholder = st.empty()

        # Optimization Parameters
        frame_skip = 5 # Process every 5th frame
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
                    
                    texture_map = generate_texture_heatmap(gray, method=method_key, win=v_win, step=v_step, levels=glcm_levels)
                    
                    heatmap_uint8 = (texture_map * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))
                    
                    processed_display = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                except:
                    pass

            # Update UI
            video_placeholder.image(processed_display, channels="BGR", use_container_width=True)
            
            end_time = time.time()
            fps = 1.0 / (end_time - start_time + 1e-6)
            fps_placeholder.metric("Processing FPS", f"{fps:.1f}")
            
            # Extract features for current frame
            try:
                current_feats = extract_all_features(gray, levels=glcm_levels)
                feat_df_vid = pd.DataFrame(current_feats.items(), columns=["Feature", "Value"])
                feats_placeholder.dataframe(feat_df_vid, height=300)
            except:
                pass
                
            time.sleep(0.01)
            
        cap.release()


