
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
 
# ----------------------------------------------------------------------
# Page settings
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Fruit Detection using YOLOv8",
    page_icon="🍉",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ----------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
 
    @keyframes drift {
        0%   { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 50% 50%; }
        50%  { background-position: 10% 10%, 90% 15%, 15% 85%, 85% 90%, 45% 55%; }
        100% { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 50% 50%; }
    }
    @keyframes float-slow {
        0%   { transform: translateY(0px) rotate(0deg); }
        50%  { transform: translateY(-18px) rotate(8deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
 
    /* Overall app background: soft pastel mesh */
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(255, 179, 186, 0.25) 0%, transparent 40%),
            radial-gradient(circle at 85% 15%, rgba(255, 223, 186, 0.28) 0%, transparent 42%),
            radial-gradient(circle at 20% 85%, rgba(186, 255, 201, 0.28) 0%, transparent 42%),
            radial-gradient(circle at 85% 88%, rgba(186, 225, 255, 0.28) 0%, transparent 42%),
            radial-gradient(circle at 50% 50%, rgba(223, 186, 255, 0.20) 0%, transparent 55%),
            linear-gradient(160deg, #fdfcfb 0%, #f7f9fc 100%);
        background-size: 180% 180%, 180% 180%, 180% 180%, 180% 180%, 160% 160%, 100% 100%;
        animation: drift 22s ease-in-out infinite;
    }
 
    /* Decorative floating fruit emojis, purely CSS/HTML, no external images */
    .fruit-deco {
        position: fixed;
        font-size: 3.2rem;
        opacity: 0.16;
        pointer-events: none;
        z-index: 0;
        filter: blur(0.3px);
        animation: float-slow 7s ease-in-out infinite;
    }
 
    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
 
    /* Make sure real content sits above the decorative layer */
    .block-container { position: relative; z-index: 1; }
 
    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #ff8fa3, #ffb15c, #f2c94c, #4fd1a5, #5c9dff, #a78bfa, #ff8fa3);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: drift 8s linear infinite;
    }
    .hero p {
        color: #5c4a3d;
        font-size: 1.08rem;
        font-weight: 500;
        margin-top: 0;
    }
 
    /* Card containers */
    .card {
        background: rgba(255, 255, 255, 0.88);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 28px rgba(120, 70, 40, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .card h3 {
        margin-top: 0;
        color: #3d2b1f;
    }
 
    /* Metric pills — each one its own fruity color */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .metric-pill {
        flex: 1;
        min-width: 140px;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        text-align: center;
        box-shadow: 0 6px 16px rgba(120, 70, 40, 0.16);
        color: white;
    }
    .metric-pill:nth-child(1) { background: linear-gradient(135deg, #ffb3ba, #ffd8a8); color: #5c3a3a; }
    .metric-pill:nth-child(2) { background: linear-gradient(135deg, #ffe8a3, #ffd8a8); color: #5c4a2a; }
    .metric-pill:nth-child(3) { background: linear-gradient(135deg, #bdf5d3, #9be8c4); color: #2a5c40; }
    .metric-pill:nth-child(4) { background: linear-gradient(135deg, #bcd9ff, #cdbfff); color: #333366; }
    .metric-pill .num {
        font-size: 1.7rem;
        font-weight: 800;
    }
    .metric-pill .label {
        font-size: 0.8rem;
        opacity: 0.92;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
 
    /* Detection chips */
    .chip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: white;
        border-radius: 12px;
        padding: 0.55rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 3px 10px rgba(120, 70, 40, 0.10);
        border-left: 7px solid #ff922b;
        transition: transform 0.15s ease;
    }
    .chip:hover { transform: translateX(4px); }
    .chip .name {
        font-weight: 700;
        color: #3d2b1f;
        text-transform: capitalize;
    }
    .chip .conf {
        font-weight: 700;
        color: #2a5c40;
        background: linear-gradient(135deg, #bdf5d3, #9be8c4);
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
    }
 
    /* Upload box */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 18px;
        padding: 1rem;
        border: 3px dashed #cdbfff;
    }
 
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dbe9ff 0%, #e6dcff 45%, #d6f7e8 100%);
    }
    section[data-testid="stSidebar"] code {
        background: rgba(255,255,255,0.7);
        border-radius: 6px;
    }
 
    /* Slider accent colors */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #cdbfff !important;
    }
 
    /* Footer note */
    .footer-note {
        text-align: center;
        color: #6b5d54;
        font-size: 0.82rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }
 
</style>
 
<div class="fruit-deco" style="top: 6%; left: 3%; animation-delay: 0s;">🍓</div>
<div class="fruit-deco" style="top: 12%; right: 5%; animation-delay: 1.2s;">🍍</div>
<div class="fruit-deco" style="top: 45%; left: 1%; animation-delay: 2.1s;">🍒</div>
<div class="fruit-deco" style="bottom: 8%; left: 6%; animation-delay: 0.6s;">🍉</div>
<div class="fruit-deco" style="bottom: 15%; right: 4%; animation-delay: 1.8s;">🥭</div>
<div class="fruit-deco" style="top: 65%; right: 2%; animation-delay: 3s;">🍇</div>
<div class="fruit-deco" style="top: 30%; left: 45%; opacity: 0.08; font-size: 5rem; animation-delay: 2.4s;">🍑</div>
""", unsafe_allow_html=True)
 
# ----------------------------------------------------------------------
# Fixed, class-based colors for consistent chip borders
# ----------------------------------------------------------------------
CLASS_COLORS = {
    "pineapple": "#ffd8a8",
    "cherry": "#ffb3ba",
    "mango": "#ffdca8",
    "plum": "#d8bfff",
    "tomato": "#ffbfa8",
    "watermelon": "#a8e8bf",
}
 
 
@st.cache_resource
def load_model():
    return YOLO("best.pt")
 
 
model = load_model()
 
# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 1.0, 0.5, 0.05)
    iou_threshold = st.slider(
        "Overlap (IoU) threshold", 0.05, 1.0, 0.4, 0.05,
        help="Lower = merges/removes more overlapping duplicate boxes on the same object."
    )
 
    st.markdown("---")
    st.markdown("### 📦 About this model")
    st.markdown(
        f"""
        A **YOLOv8** object detector fine-tuned to recognize
        **{len(model.names)} fruit classes**:
 
        {", ".join(f"`{n}`" for n in model.names.values())}
        """
    )
 
    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown(
        """
        1. Upload a photo containing fruit
        2. If you see duplicate boxes on the same fruit, lower the **overlap threshold**
        3. If you see missed or weak detections, lower the **confidence threshold**
        4. View detected fruits with bounding boxes & confidence scores
        """
    )
 
# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero card">
        <h1>🍇 Fruit Detection using YOLOv8</h1>
        <p>Upload an image and let a YOLOv8 model find and label the fruits in it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ----------------------------------------------------------------------
# Upload
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drop an image here, or click to browse",
    type=["jpg", "jpeg", "png"],
)
 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
 
    with st.spinner("🔍 Detecting fruits..."):
        start = time.time()
        results = model.predict(source=image_array, conf=conf_threshold, iou=iou_threshold)
        elapsed = time.time() - start
 
    result = results[0]
    annotated = result.plot()
 
    col1, col2 = st.columns(2, gap="large")
 
    with col1:
        st.markdown('<div class="card"><h3>📷 Original Image</h3>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with col2:
        st.markdown('<div class="card"><h3>🎯 Detection Result</h3>', unsafe_allow_html=True)
        st.image(annotated, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    # Metrics row
    num_detections = len(result.boxes)
    unique_classes = len({int(box.cls) for box in result.boxes}) if num_detections else 0
    avg_conf = (
        f"{float(np.mean([float(b.conf) for b in result.boxes])):.0%}"
        if num_detections else "—"
    )
 
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-pill"><div class="num">{num_detections}</div><div class="label">Objects found</div></div>
            <div class="metric-pill"><div class="num">{unique_classes}</div><div class="label">Fruit types</div></div>
            <div class="metric-pill"><div class="num">{avg_conf}</div><div class="label">Avg. confidence</div></div>
            <div class="metric-pill"><div class="num">{elapsed:.2f}s</div><div class="label">Inference time</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    # Detected objects list
    st.markdown('<div class="card"><h3> Detected Objects</h3>', unsafe_allow_html=True)
 
    if num_detections == 0:
        st.warning("No objects detected — try lowering the confidence threshold in the sidebar.")
    else:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = model.names[cls]
            color = CLASS_COLORS.get(name.lower(), "#ff922b")
            st.markdown(
                f"""
                <div class="chip" style="border-left-color: {color};">
                    <span class="name"> {name}</span>
                    <span class="conf">{conf:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
 
    st.markdown("</div>", unsafe_allow_html=True)
 
else:
    st.info("👆 Upload an image above to get started.")
 
st.markdown(
    '<div class="footer-note">Built with YOLOv8 + Streamlit</div>',
    unsafe_allow_html=True,
)