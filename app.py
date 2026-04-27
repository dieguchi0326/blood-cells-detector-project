import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CONF = 0.15
IOU = 0.7
IMGSZ = 640

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}
# Switch BGR colors from predict.py to RGB for Streamlit/Pillow
COLOR_RBC = (255, 0, 0)
COLOR_PLATELETS = (0, 200, 0)
COLOR_WBC = (0, 80, 255)

def color_for(name: str) -> tuple[int, int, int]:
    if name == "RBC":
        return COLOR_RBC
    if name == "Platelets":
        return COLOR_PLATELETS
    if name in WBC_SUBTYPES:
        return COLOR_WBC
    return (200, 200, 200)

def annotate(img, boxes_xyxy, classes, names) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft, bt = 0.4, 1, 1
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, classes):
        name = names[int(c)]
        col = color_for(name)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, bt)
        (tw, th), _ = cv2.getTextSize(name, font, fs, ft)
        ly = y1 - 2
        if ly - th - 2 < 0:
            ly = y1 + th + 4
        # Note: the rectangle for background needs to be filled (-1), cv2 colors expect BGR if we are not careful
        # But we are drawing on an RGB image, so everything is RGB
        cv2.rectangle(img, (x1, ly - th - 2), (x1 + tw + 2, ly + 1), col, -1)
        cv2.putText(img, name, (x1 + 1, ly - 1), font, fs,
                    (255, 255, 255), ft, cv2.LINE_AA)

@st.cache_resource
def load_model():
    model_path = "blood_detector_model.pt"
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

st.title("🩸 Blood Cell Detector")
st.markdown("Upload a blood smear image and let the model detect Red Blood Cells, Platelets, and White Blood Cells!")

model = load_model()

if model is None:
    st.error("Model file `blood_detector_model.pt` not found in the repository. Please make sure it's uploaded.")
else:
    st.sidebar.title("Configuration")
    st.sidebar.markdown("Adjust inference parameters:")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, CONF, 0.01)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, IOU, 0.01)

    uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Show spinner during processing
        with st.spinner('Analyzing image...'):
            image = Image.open(uploaded_file).convert("RGB")
            # Convert to numpy array for inference and OpenCV drawing
            img_np = np.array(image)

            # Inference
            results = model.predict(
                source=img_np,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=IMGSZ,
                save=False,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names

            # Draw bounding boxes
            img_annotated = img_np.copy()
            annotate(img_annotated, boxes, classes, names)

            # Display side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detected Cells")
                st.image(img_annotated, use_column_width=True)

            # Show cell counts
            st.divider()
            st.subheader("Detection Summary")
            counts = Counter(names[int(c)] for c in classes)
            
            if len(boxes) == 0:
                st.info("No cells detected.")
            else:
                st.success(f"Total cells detected: **{len(boxes)}**")
                
                # Display metrics neatly
                metric_cols = st.columns(len(counts))
                for idx, (cell_type, count) in enumerate(counts.items()):
                    with metric_cols[idx % len(metric_cols)]:
                        st.metric(label=cell_type, value=count)
