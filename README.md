# 🎨 AI Visual Intelligence System

An AI-powered visual understanding application that analyzes uploaded images using image captioning, object detection, facial-expression estimation, scene analysis, and grounded creative storytelling.

## 🚀 Live Demo

**Render:** https://ai-visual-intelligence-system.onrender.com

> The first request can take longer because the service may need to wake up and load/download ML models.

## ✨ Features

- **Automatic Image Captioning** using Salesforce BLIP
- **Object Detection** using YOLOv8
- **Human Facial-Expression Estimation** using DeepFace
- **Environment / Scene Estimation** with conservative visual reasoning
- **Deep Scene Description** combining caption, detected objects, lighting, setting, and expression output
- **Creative Story Generation** using FLAN-T5 with a grounded fallback story system
- **Interactive Gradio UI** with image upload and structured outputs

## 🧠 Tech Stack

**Language:** Python  
**UI:** Gradio  
**API / App Layer:** FastAPI  
**Image Captioning:** Salesforce BLIP  
**Object Detection:** Ultralytics YOLOv8  
**Facial Analysis:** DeepFace, OpenCV  
**Text Generation:** Google FLAN-T5  
**ML Frameworks:** PyTorch, TensorFlow  
**Deployment:** Render  
**Version Control:** Git, GitHub

## 🔍 How It Works

1. The user uploads an image.
2. BLIP generates an image caption.
3. YOLOv8 detects visible objects and identifies person regions.
4. If a clear human face is detected, DeepFace estimates the facial expression.
5. The app estimates the visual environment without claiming an exact location when evidence is weak.
6. A structured scene description is generated from the available visual evidence.
7. FLAN-T5 attempts to generate a short image-grounded story. If the result is too short or poorly grounded, the app uses a deterministic fallback story.

## ⚠️ Important Note

Facial-expression output is an automated visual estimate only. It should not be treated as a person's confirmed emotional state.

The environment output is also intentionally conservative and avoids claiming an exact real-world location unless the image provides sufficient visual evidence.

## 💻 Run Locally

Use Python 3.11.

```bash
git clone https://github.com/niyatisingh2809/ai-visual-intelligence-system.git
cd ai-visual-intelligence-system

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

The first analysis may take extra time because the required models are downloaded and loaded on demand.

## 📂 Main Project Files

```text
ai-visual-intelligence-system/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── tests/
│   └── test_app.py
└── README.md
```

## 🧪 Testing

```bash
python -m unittest discover -s tests -v
```

## 🌐 Deployment

The application is deployed as a Python web service on Render.

**Live Application:**  
https://ai-visual-intelligence-system.onrender.com

The app binds to `0.0.0.0` and uses the platform-provided `PORT` environment variable when deployed.

## 📌 Current Model Note

The main deployed application currently uses **Salesforce BLIP image captioning base** for caption generation together with YOLOv8, FLAN-T5, and DeepFace.

## 👩‍💻 Author

**Niyati Singh**

GitHub: https://github.com/niyatisingh2809
