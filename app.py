import streamlit as st
import cv2
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------------
# Load COCO class names
# -----------------------------
classNames = []
with open("coco.names", "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# -----------------------------
# Load SSD Mobilenet Model
# -----------------------------
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.title("⚙ Settings")

confThreshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

show_fps = st.sidebar.checkbox("Show FPS", True)
show_labels = st.sidebar.checkbox("Show Labels", True)
show_conf = st.sidebar.checkbox("Show Confidence %", True)

# Color picker for bounding boxes
box_color = st.sidebar.color_picker("Box Color", "#00FF00")  # default green
box_color_rgb = tuple(int(box_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))


# -----------------------------
# Main Title
# -----------------------------
st.title("🚀 Advanced Object Detection System (Real-Time + Image Upload)")
st.write("Using **SSD MobileNet v3 + Streamlit**")

# -----------------------------
# Webcam Detection Class
# -----------------------------
class ObjectDetection(VideoTransformerBase):
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        classIds, confs, bboxes = net.detect(img, confThreshold=confThreshold)

        # FPS Calculation (Corrected)
        curr_time = time.time()
        if show_fps:
            self.fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        # Draw detections
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bboxes):
                cv2.rectangle(img, box, box_color_rgb, 2)
                label = classNames[classId - 1]

                if show_conf:
                    label += f" ({round(confidence * 100)}%)"

                if show_labels:
                    cv2.putText(img, label, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color_rgb, 2)

        if show_fps:
            cv2.putText(img, f"FPS: {int(self.fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return img


# ---------------------------------------------------------
# WebRTC LIVE CAMERA SECTION
# ---------------------------------------------------------
st.subheader("📸 Live Webcam Object Detection")

webrtc_streamer(
    key="object-detection",
    video_transformer_factory=ObjectDetection,
    media_stream_constraints={"video": True, "audio": False}
)


# ---------------------------------------------------------
# IMAGE UPLOAD SECTION
# ---------------------------------------------------------
st.subheader("🖼 Upload Image for Detection")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    classIds, confs, bboxes = net.detect(img, confThreshold=confThreshold)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bboxes):
            cv2.rectangle(img, box, box_color_rgb, 2)

            label = classNames[classId - 1]
            if show_conf:
                label += f" ({round(confidence * 100)}%)"

            if show_labels:
                cv2.putText(img, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color_rgb, 2)

    st.image(img, channels="BGR", caption="Detected Image")

    result, encoded = cv2.imencode(".jpg", img)
    st.download_button("💾 Download Result", encoded.tobytes(),
                       file_name="detected_image.jpg", mime="image/jpeg")
    