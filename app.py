import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Underwater Segmentation Demo",
    page_icon="🌊",
    layout="wide"
)

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_all_models():
    return {
        "Baseline (SegFormer)": torch.jit.load("segformer.pt", map_location="cpu"),
        "Proposed (MiT-B0)": torch.jit.load("model.pt", map_location="cpu"),
        "Proposed (MiT-B1)": torch.jit.load("proposed_mitb1.pt", map_location="cpu")
    }

models = load_all_models()

for m in models.values():
    m.eval()

# ---------------------------
# HEADER (LOGO + TITLE)
# ---------------------------
# ---------------------------
# CENTERED HEADER
# ---------------------------
# Center logo separately
c1, c2, c3, c4, c5 = st.columns([1,2,2,1,1])

with c3:
    st.image("logo.PNG", width=120)

# Center text using HTML
st.markdown("""
<div style="text-align: center; line-height:1.6;">

<h4 style="margin-bottom:5px;">Indian Institute of Technology Goa</h4>

<h5 style="margin-top:0;">M.Tech Thesis Demonstration</h5>

<h3 style="margin:15px 0;"><b>Underwater Image Segmentation using Transformer-based Architecture with UAFM</b></h3>

<p><b>Presented by:</b> Manvi Jain</p>
<p><b>Advisor:</b> Dr. Shitala Prasad</p>
<p><b>Department:</b> Computer Science and Engineering</p>

</div>
""", unsafe_allow_html=True)




st.divider()

# ---------------------------
# ABOUT SECTION
# ---------------------------
with st.expander("📘 About This Work"):
    st.write("""
    Underwater image segmentation is a challenging task due to visibility degradation, wavelength-dependent light absorption, color distortion, and scattering effects in aquatic environments. To address these issues, we propose an enhanced transformer-based segmentation framework built upon the SegFormer architecture, integrating a lightweight Underwater Adaptive Feature Modulation (UAFM) module to improve feature representation under degraded conditions. The UAFM module jointly models channel-wise and spatial contextual dependencies and adaptively fuses them via learnable weighting parameters, enabling effective feature recalibration after multi-scale feature aggregation with minimal computational overhead. Extensive experiments conducted on the SUIM benchmark dataset demonstrate that the proposed model achieves a mean Intersection over Union (mIoU) of 81.11\%, outperforming the baseline SegFormer (80.52\%). Furthermore, compared to parameter-intensive models such as UW-SegFormer (21.78M parameters), the proposed approach reduces model complexity by approximately 84\% while maintaining competitive or superior segmentation performance. These results show that the proposed framework achieves an effective balance between segmentation accuracy and computational efficiency, making it suitable for real-time, resource-constrained underwater applications.

    This application demonstrates underwater semantic segmentation using Transformer-based architectures.

    **Models Included:**
    - Baseline: SegFormer  
    - Proposed: MiT B0 and MiT b1

    **Challenges Addressed:**
    - Low visibility
    - Color distortion
    - Light scattering
    - Blurred boundaries
    """)

# ---------------------------
# MODEL SELECTION
# ---------------------------
st.subheader("🔍 Choose Model")

model_option = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[model_option]

# ---------------------------
# IMAGE UPLOAD
# ---------------------------
st.subheader("📷 Upload Image")

uploaded_file = st.file_uploader("Upload an underwater image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # ---------------------------
    # READ IMAGE
    # ---------------------------
    image = Image.open(uploaded_file).convert("RGB")

    # ---------------------------
    # PREPROCESSING
    # ---------------------------
    img = np.array(image)

    img_resized = cv2.resize(img, (640, 480))
    img_resized = img_resized / 255.0
    img_resized = img_resized.transpose(2, 0, 1)

    img_tensor = torch.tensor(img_resized).float().unsqueeze(0)

    # ---------------------------
    # PREDICTION
    # ---------------------------
    with torch.no_grad():
        output = model(img_tensor)

    pred = output.argmax(dim=1).squeeze().numpy()

    # ---------------------------
    # COLOR MAP
    # ---------------------------
    def color_map(mask):
        colors = np.array([
            [0, 0, 0],       # Background
            [0, 0, 255],     # Diver
            [0, 255, 255],   # Wreck
            [255, 0, 0],     # Robot
            [255, 0, 255],   # Reef
            [255, 255, 0]    # Fish
        ])
        return colors[mask % len(colors)].astype(np.uint8)

    colored_mask = color_map(pred)

    # resize back
    colored_mask = cv2.resize(colored_mask, (image.size[0], image.size[1]))

    # ---------------------------
    # DISPLAY
    # ---------------------------
    st.subheader("🧾 Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image", width=500)

    with col2:
        st.image(colored_mask, caption="Segmented Output", width=500)

    # ---------------------------
    # DETECTED OBJECTS
    # ---------------------------
    classes = {
        0: "Background Water",
        1: "Human Diver",
        2: "Wrecks/Ruins",
        3: "Robots/Instruments",
        4: "Reefs/Invertebrates",
        5: "Fish/Vertebrates"
    }

    unique_classes = np.unique(pred)
    detected = [classes[int(cls)] for cls in unique_classes if int(cls) in classes]

    st.subheader("🧠 Detected Objects")

    if len(detected) > 0:
        for item in detected:
            st.markdown(f"- {item}")
    else:
        st.warning("No known objects detected.")

# ---------------------------
# FOOTER
# ---------------------------
st.divider()
st.markdown("""
<center>
Developed as part of M.Tech Thesis (2026)  
Indian Institute of Technology Goa
</center>
""", unsafe_allow_html=True)