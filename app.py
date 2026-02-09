
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
    
    # input_source = st.selectbox("Input Source", ["Upload Video", "Webcam"]) 
    # For now, simplifying as per user request to focus on Image first
    mode = st.radio("Mode", ["Single Image Analysis", "Video/Real-time"])
    
    st.markdown("### Analysis Parameters")
    glcm_levels = st.slider("GLCM Levels", 8, 256, 32, step=8)
    
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
    
    # Agentic Upload Area - Centered and Clean
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], help="Drop a colonoscopy frame here", label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Read Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Layout: Image on Left, Analysis on Right
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, channels="BGR", caption="Uploaded Frame", use_container_width=True)
            # Keeping use_container_width=True for now as it works in 1.x without error, just warning. 
            # width="stretch" is for 1.40+ or so.
            
        with col2:
            st.markdown("### Texture Analysis")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # --- Heatmap Overlay Control ---
            st.markdown("#### Heatmap Overlay")
            heatmap_method = st.selectbox("Method", ["None", "Local Entropy (Complexity)", "Local Std Dev (Roughness)"])
            
            overlay_img = None
            if heatmap_method != "None":
                alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.4, 0.05)
                
                method_key = 'entropy' if "Entropy" in heatmap_method else 'std'
                with st.spinner(f"Generating {heatmap_method}..."):
                    try:
                        from features import generate_texture_heatmap
                        texture_map_norm = generate_texture_heatmap(gray, method=method_key)
                        
                        # Apply colormap
                        heatmap_uint8 = (texture_map_norm * 255).astype(np.uint8)
                        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                        
                        # Resize heatmap to match original image if needed (should be same size but good practice)
                        if heatmap_color.shape[:2] != image.shape[:2]:
                            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
                            
                        # Blend
                        overlay_img = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
                        
                        st.image(overlay_img, channels="BGR", caption=f"{heatmap_method} Overlay", use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating heatmap: {e}")

            st.markdown("---")
            
            with st.spinner("Extracting Features..."):
                try:
                    feats = extract_all_features(gray, levels=glcm_levels)
                    
                    # Display Top Features
                    feat_df = pd.DataFrame(feats.items(), columns=["Feature", "Value"])
                    st.dataframe(feat_df, height=200)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

        # Full Width Visualizations
        st.markdown("---")
        st.markdown("### Feature Distribution Heatmap")
        
        # Heatmap
        current_vector = np.array(list(feats.values()))
        z_scores = (current_vector - np.mean(current_vector)) / (np.std(current_vector) + 1e-5) # Simple self-normalization for viz
        
        z_grid = z_scores.reshape(4, 5)
        fig, ax = plt.subplots(figsize=(10, 3))
        # Dark background for plot
        fig.patch.set_facecolor('#222831')
        ax.set_facecolor('#222831')
        
        # Custom annotations
        sns.heatmap(z_grid, cmap='viridis', center=0, annot=True, fmt=".1f", cbar=True, ax=ax, 
                    yticklabels=False, xticklabels=False)
        
        # Update text color for dark mode
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        st.pyplot(fig)
        plt.close(fig)

elif mode == "Video/Real-time":
    st.warning("Video Mode is currently in beta. Switch to 'Single Image Analysis' for the stable agentic flow.")
    
    # Legacy video code (optional to keep or comment out)
    # Keeping the processing function but not calling it directly unless uploaded
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        col1, col2 = st.columns([2, 1])
        with col1:
            video_placeholder = st.empty()
        with col2:
            status_placeholder = st.empty()
            features_placeholder = st.empty()
            heatmap_placeholder = st.empty()
            
        # Re-using the process logic would require slight adaptation to not use global input_source
        # For now, let's just focus on image as requested.


