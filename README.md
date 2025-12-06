# 🧠 Object Detection App
- A simple and powerful real-time object detection system built using Python, OpenCV, and MobileNet SSD.
The app can detect multiple objects from live camera feed and label them with bounding boxes.

# 🚀 Features

- 🎥 Real-time object detection using webcam
- 🔍 MobileNet SSD (COCO Dataset) for fast & lightweight detection
- 🧩 Easy camera selection UI
- 📦 Works locally & deploys easily on Streamlit Cloud
- ⚡ Lightweight & beginner-friendly codebase

# 📁 Project Structure
- object_detection_app/
│
├── app.py                          # Main Streamlit app
├── coco.names                      # Class labels (COCO dataset)
├── frozen_inference_graph.pb       # Pretrained MobileNet SSD model
├── ssd_mobilenet_v3_large_coco...  # Model config file
├── requirements.txt                # Project dependencies
└── .devcontainer/                  # Dev container configs (Optional)

# 🛠️ Installation & Setup
- 1️⃣ Clone the repository
git clone https://github.com/CodingwithAnkit-tech/object_detection_app.git
cd object_detection_app

- 2️⃣ Install dependencies
pip install -r requirements.txt

- 3️⃣ Run the application
streamlit run app.py

# 🎯 How It Works

Inside the app:

Loads MobileNet SSD pretrained model

Opens the selected webcam

Detects objects frame-by-frame

Draws bounding boxes & class labels in real time

This makes it ideal for learning computer vision, demo projects, or college submissions.

📸 Demo Screenshot

<img width="1920" height="1080" alt="Screenshot (146)" src="https://github.com/user-attachments/assets/4283ae7d-698c-4e1b-bfee-a2aad2957cc6" />


# 📦 Requirements

- Python 3.8+

- OpenCV

- Streamlit

- Numpy

(Already included in requirements.txt)

# 🌐 Deployment on Streamlit Cloud

Push your repo to GitHub

Go to streamlit.io → Deploy app

Select your repo

Set:

Main file: app.py

Done! Your app will be live.

# Here the project link-

https://objectdetectionapp-iqtof9nhpugzp5jkv9ud4w.streamlit.app/
