import streamlit as st
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter
from model import UNet3D
from detector import TumorDetector
from gradcam import GradCAM

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
BASE_PATH = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post"
PROCESSED_PATH = r"E:\EDI\tumor_growth_project\data\processed"
MODEL_PATH = "tumor_model.pth"
DETECTOR_PATH = "detector_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Neuro-Oncology AI", layout="wide", page_icon="ðŸ§ ")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Segoe UI', sans-serif; }
    .metric-container { background-color: #262730; padding: 15px; border-radius: 8px; border: 1px solid #41444e; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 50px; 
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%); 
        color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS & LOADERS
# ==========================================
@st.cache_data
def get_patient_list():
    """Scans the PROCESSED folder for both Tumor and Healthy patients."""
    if os.path.exists(PROCESSED_PATH):
        # We check for folders that contain Timepoint subfolders
        return sorted([d for d in os.listdir(PROCESSED_PATH) if os.path.isdir(os.path.join(PROCESSED_PATH, d))])
    return []

@st.cache_resource
def load_models():
    """Loads both AI models (Detector and Predictor) into GPU/CPU memory."""
    models = {}
    
    # Load Detector (Stage 1)
    if os.path.exists(DETECTOR_PATH):
        try:
            det = TumorDetector().to(DEVICE)
            det.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
            det.eval()
            models['detector'] = det
        except Exception as e:
            st.error(f"Error loading Detector: {e}")

    # Load Predictor (Stage 2): future CT scan prediction from two timepoints
    if os.path.exists(MODEL_PATH):
        try:
            pred = UNet3D(n_channels=4, n_classes=1).to(DEVICE)
            pred.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            pred.eval()
            models['predictor'] = pred
        except Exception as e:
            st.error(f"Error loading Predictor: {e}")
            
    return models

def compute_volume(mask_path):
    """Calculates tumor volume in cmÂ³ from a binary mask."""
    if not os.path.exists(mask_path): return 0.0
    try:
        mask = np.load(mask_path)
        # Assuming isotropic 1mmÂ³ resizing in preprocess steps (128x128x64 or 64x64x32)
        # For normalized/resized data we estimate using voxel count.
        voxel_vol_mm3 = 1.0 
        return (np.sum(mask > 0) * voxel_vol_mm3) / 1000.0
    except:
        return 0.0

def load_scan_normalized(path):
    """Loads .npy scan and normalizes it to 0-1 range."""
    img = np.load(path)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
models = load_models()
patients = get_patient_list()

st.sidebar.title("ðŸ©º Neuro-Oncology AI")
st.sidebar.markdown(f"**System Status:** {'ðŸŸ¢ Online' if models else 'ðŸ”´ Offline'}")

if patients:
    selected_patient = st.sidebar.selectbox("Select Patient Record", patients)
    # Path is now primarily the PROCESSED folder
    p_dir = os.path.join(PROCESSED_PATH, selected_patient)
    timepoints = sorted([d for d in os.listdir(p_dir) if d.startswith("Timepoint_")], 
                       key=lambda x: int(x.split('_')[-1]))
else:
    selected_patient = None
    timepoints = []
    st.sidebar.warning("No processed data found. Run `preprocess.py`.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Operation Mode", ["ðŸ“Š Analysis Dashboard", "ðŸ§  AI Diagnosis & Prediction"])

# ==========================================
# 4. MODE: ANALYSIS DASHBOARD
# ==========================================
if mode == "ðŸ“Š Analysis Dashboard" and selected_patient:
    st.title(f"Patient Analysis: {selected_patient}")
    
    # Distinction: Is this a Healthy control or a Tumor patient?
    is_healthy = "Healthy" in selected_patient
    
    if is_healthy:
        # --- HEALTHY PATIENT VIEW ---
        st.success("âœ… **Patient identified as Healthy Reference (IXI Dataset).**")
        st.info("This record is used as a negative control for training the Tumor Detector.")
        
        col1, col2 = st.columns(2)
        col1.metric("Tumor Volume", "0.00 cmÂ³", delta="Normal")
        col1.metric("Risk Factor", "Low", delta="-100%")
        
        # Healthy Viewer
        if timepoints:
            scan_path = os.path.join(p_dir, timepoints[0], "scan.npy")
            if os.path.exists(scan_path):
                scan = load_scan_normalized(scan_path)
                slice_idx = col2.slider("Axial Slice View", 0, scan.shape[2]-1, scan.shape[2]//2)
                
                fig, ax = plt.subplots(figsize=(4,4))
                ax.imshow(np.rot90(scan[:, :, slice_idx]), cmap="gray")
                ax.axis('off')
                fig.patch.set_facecolor('#0e1117')
                col2.pyplot(fig)
                
    else:
        # --- TUMOR PATIENT VIEW ---
        growth_data = []
        tp_labels = []
        
        # Calculate volumes
        for tp in timepoints:
            mask_path = os.path.join(p_dir, tp, "mask.npy")
            vol = compute_volume(mask_path)
            growth_data.append(vol)
            tp_labels.append(tp)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Longitudinal Scans", len(timepoints))
        c2.metric("Latest Volume", f"{growth_data[-1]:.2f} cmÂ³" if growth_data else "0.00 cmÂ³")
        
        growth_pct = 0.0
        if len(growth_data) > 1 and growth_data[0] > 0:
            growth_pct = ((growth_data[-1] - growth_data[0]) / growth_data[0]) * 100
        c3.metric("Total Growth", f"{growth_pct:.1f}%", delta_color="inverse")

        # Growth Chart
        st.subheader("Tumor Growth Trajectory")
        if len(growth_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(tp_labels, growth_data, color='#FF4B4B', marker='o', linewidth=2)
            ax.fill_between(tp_labels, growth_data, color='#FF4B4B', alpha=0.1)
            ax.set_ylabel("Volume (cmÂ³)")
            ax.grid(True, linestyle='--', alpha=0.2)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            for spine in ax.spines.values(): spine.set_edgecolor('#41444e')
            st.pyplot(fig)

        # --- 3D SLICER (UPGRADED) ---
        st.subheader("MRI Slice Inspection (3D Slicer)")
        if timepoints:
            selected_tp = st.selectbox("Select Timepoint", timepoints)
            
            scan_path = os.path.join(p_dir, selected_tp, "scan.npy")
            mask_path = os.path.join(p_dir, selected_tp, "mask.npy")
            
            if os.path.exists(scan_path) and os.path.exists(mask_path):
                scan = load_scan_normalized(scan_path)
                mask = np.load(mask_path)
                
                # --- Multi-View Logic ---
                view = st.radio("Select View Plane", ["Axial (Top)", "Coronal (Front)", "Sagittal (Side)"], horizontal=True)
                
                # Slicing logic based on dimensions (64, 64, 32)
                # Axial: Z-axis (2), Coronal: Y-axis (1), Sagittal: X-axis (0)
                if view == "Axial (Top)":
                    max_slice = scan.shape[2] - 1
                    slice_idx = st.slider("Slice Index", 0, max_slice, max_slice//2)
                    img_slice = scan[:, :, slice_idx]
                    mask_slice = mask[:, :, slice_idx]
                elif view == "Coronal (Front)":
                    max_slice = scan.shape[1] - 1
                    slice_idx = st.slider("Slice Index", 0, max_slice, max_slice//2)
                    img_slice = scan[:, slice_idx, :]
                    mask_slice = mask[:, slice_idx, :]
                else: # Sagittal
                    max_slice = scan.shape[0] - 1
                    slice_idx = st.slider("Slice Index", 0, max_slice, max_slice//2)
                    img_slice = scan[slice_idx, :, :]
                    mask_slice = mask[slice_idx, :, :]

                # Rotation Fixer
                # rot = st.slider("Adjust Rotation", 0, 270, 90, 90)
                img_slice = np.rot90(img_slice, k=1)
                mask_slice = np.rot90(mask_slice, k=1)

                # Plot
                fig_s, ax_s = plt.subplots(figsize=(6,6))
                # aspect='auto' prevents squashed images for non-axial views
                ax_s.imshow(img_slice, cmap="gray", aspect='auto') 
                
                # Overlay mask
                m = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax_s.imshow(m, cmap="autumn", alpha=0.6, aspect='auto')
                
                ax_s.axis('off')
                fig_s.patch.set_facecolor('#0e1117')
                st.pyplot(fig_s)

# ==========================================
# 5. MODE: AI DIAGNOSIS & PREDICTION
# ==========================================
elif mode == "🧠 AI Diagnosis & Prediction" and selected_patient:
    st.title("🤖 AI Diagnostic Pipeline")

    if len(timepoints) < 3:
        st.error("Future CT scan prediction needs at least 3 timepoints: two inputs (T1, T2) and one future target (T3).")
    else:
        # Input selection: two consecutive timepoints -> predict next future scan
        max_start = len(timepoints) - 3
        start_idx = st.selectbox(
            "Select input pair start (T1, T2 -> predict T3)",
            list(range(max_start + 1)),
            format_func=lambda i: f"{timepoints[i]} + {timepoints[i+1]} -> {timepoints[i+2]}",
        )
        tp_in1 = timepoints[start_idx]
        tp_in2 = timepoints[start_idx + 1]
        tp_target = timepoints[start_idx + 2]
        st.info(f"Inputs: {tp_in1}, {tp_in2}  |  Predicting future scan: {tp_target}")

        use_dummy = st.checkbox("Test with empty brain input (simulate healthy)")

        if st.button("RUN DIAGNOSIS"):
            # Load two input timepoints (scan + mask each)
            scan1_path = os.path.join(p_dir, tp_in1, "scan.npy")
            mask1_path = os.path.join(p_dir, tp_in1, "mask.npy")
            scan2_path = os.path.join(p_dir, tp_in2, "scan.npy")
            mask2_path = os.path.join(p_dir, tp_in2, "mask.npy")
            scan_target_path = os.path.join(p_dir, tp_target, "scan.npy")

            required = [scan1_path, mask1_path, scan2_path, mask2_path]
            missing = [p for p in required if not os.path.exists(p)]
            if missing:
                st.error(f"Missing required input files: {missing[0]}")
                st.stop()

            scan1 = load_scan_normalized(scan1_path)
            mask1 = np.load(mask1_path)
            scan2 = load_scan_normalized(scan2_path)
            mask2 = np.load(mask2_path)
            target_scan = load_scan_normalized(scan_target_path) if os.path.exists(scan_target_path) else None

            # --- STAGE 1: DETECTION (on latest input scan) ---
            st.markdown("### Stage 1: Tumor Detection")
            if "detector" not in models:
                st.error("Detector model not found. Train it first (detector_model.pth).")
                st.stop()

            if use_dummy:
                st.warning("TESTING MODE: Feeding empty brain volume to detector...")
                det_input = torch.zeros(1, 1, scan2.shape[0], scan2.shape[1], scan2.shape[2]).to(DEVICE)
            else:
                det_input = torch.from_numpy(scan2).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            target_layer = models["detector"].features[12]
            cam_extractor = GradCAM(models["detector"], target_layer)
            with st.spinner("Analyzing latest input scan and generating heatmap..."):
                prob = models["detector"](det_input).item()
                heatmap = cam_extractor.generate_cam(det_input)

            d_col1, d_col2 = st.columns([1, 4])
            d_col1.metric("Probability", f"{prob * 100:.2f}%")

            if prob < 0.5:
                d_col2.success("Result: Normal. No tumor detected by detector.")
                st.info("Pipeline terminated. Future CT scan prediction skipped.")
                st.stop()

            d_col2.error("Result: Tumor detected. Running future CT scan prediction.")

            # Grad-CAM visualization on latest input scan
            st.subheader("Explainable AI (Grad-CAM, latest input)")
            max_slice = scan2.shape[2] // 2 if use_dummy else np.unravel_index(np.argmax(heatmap), heatmap.shape)[2]
            scan_slice = np.rot90(scan2[:, :, max_slice])
            hm_smooth = gaussian_filter(np.rot90(heatmap[:, :, max_slice]), sigma=2)
            if np.max(hm_smooth) > 0:
                hm_smooth = (hm_smooth - np.min(hm_smooth)) / (np.max(hm_smooth) - np.min(hm_smooth) + 1e-8)

            gc1, gc2 = st.columns(2)
            with gc1:
                st.write(f"Latest input scan ({tp_in2}), slice {max_slice}")
                f, ax = plt.subplots()
                ax.imshow(scan_slice, cmap="gray")
                ax.axis("off")
                f.patch.set_facecolor("#0e1117")
                st.pyplot(f)
            with gc2:
                st.write("AI attention heatmap")
                f, ax = plt.subplots()
                ax.imshow(scan_slice, cmap="gray", alpha=0.5)
                ax.imshow(hm_smooth, cmap="jet", alpha=0.6, vmin=0, vmax=1)
                ax.axis("off")
                f.patch.set_facecolor("#0e1117")
                st.pyplot(f)

            # --- STAGE 2: FUTURE CT SCAN PREDICTION ---
            st.markdown("---")
            st.markdown("### Stage 2: Future CT Scan Prediction (2 timepoints -> future scan)")
            if "predictor" not in models:
                st.error("Predictor model not found. Train it first (tumor_model.pth).")
                st.stop()

            def _to_dhw(vol):
                a = np.asarray(vol)
                if a.ndim == 4:
                    a = a.squeeze()
                return np.transpose(a, (2, 0, 1))

            with st.spinner("Predicting future CT scan from two input timepoints..."):
                scan1_t = torch.from_numpy(_to_dhw(scan1)).float()
                mask1_t = torch.from_numpy(_to_dhw(mask1)).float()
                scan2_t = torch.from_numpy(_to_dhw(scan2)).float()
                mask2_t = torch.from_numpy(_to_dhw(mask2)).float()
                input_tensor = torch.stack([scan1_t, mask1_t, scan2_t, mask2_t], dim=0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    pred_scan_dhw = models["predictor"](input_tensor).clamp(0.0, 1.0).cpu().numpy()[0, 0]

            # Convert prediction back to saved (H,W,D) orientation for display
            pred_scan = np.transpose(pred_scan_dhw, (1, 2, 0))
            disp_slice = int(max_slice)
            disp_slice = max(0, min(disp_slice, pred_scan.shape[2] - 1))

            def show_scan(vol_hwd, title):
                f, ax = plt.subplots()
                ax.imshow(np.rot90(vol_hwd[:, :, disp_slice]), cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, color="white", fontsize=10)
                ax.axis("off")
                f.patch.set_facecolor("#0e1117")
                st.pyplot(f)

            if target_scan is not None:
                target_dhw = _to_dhw(target_scan)
                mae = float(np.mean(np.abs(pred_scan_dhw - target_dhw)))
                mse = float(np.mean((pred_scan_dhw - target_dhw) ** 2))
                psnr = float(20 * np.log10(1.0 / np.sqrt(mse + 1e-8)))
                st.success(f"Future scan prediction complete. MAE: {mae:.6f} | MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

                cols = st.columns(4)
                with cols[0]:
                    show_scan(scan1, f"Input 1 ({tp_in1})")
                with cols[1]:
                    show_scan(scan2, f"Input 2 ({tp_in2})")
                with cols[2]:
                    show_scan(target_scan, f"Actual Future ({tp_target})")
                with cols[3]:
                    show_scan(pred_scan, f"Predicted Future ({tp_target})")
            else:
                st.warning(f"Actual future scan not found for {tp_target}; showing prediction only.")
                cols = st.columns(3)
                with cols[0]:
                    show_scan(scan1, f"Input 1 ({tp_in1})")
                with cols[1]:
                    show_scan(scan2, f"Input 2 ({tp_in2})")
                with cols[2]:
                    show_scan(pred_scan, f"Predicted Future ({tp_target})")

            st.caption("Note: This predicts a future CT/MRI scan volume from two prior timepoints (scan+mask each). Retrain with updated src/train.py for compatible weights.")
