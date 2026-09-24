import base64
import io
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

app = FastAPI(title="AI Visual Intelligence Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMOTION_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)
EMOTION_MODEL_PATH = Path("/tmp/emotion-ferplus-8.onnx")

EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


@lru_cache(maxsize=1)
def object_detector():
    from ultralytics import YOLO

    return YOLO("yolov8n.pt")


@lru_cache(maxsize=1)
def face_detector():
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("OpenCV face detector could not be loaded.")
    return detector


def _download_emotion_model():
    if EMOTION_MODEL_PATH.exists():
        return EMOTION_MODEL_PATH

    response = requests.get(
        EMOTION_MODEL_URL,
        timeout=90,
        allow_redirects=True,
    )
    response.raise_for_status()
    EMOTION_MODEL_PATH.write_bytes(response.content)
    return EMOTION_MODEL_PATH


@lru_cache(maxsize=1)
def emotion_net():
    model_path = _download_emotion_model()
    return cv2.dnn.readNetFromONNX(str(model_path))


def visual_details(image):
    pixels = np.asarray(
        image.convert("RGB").resize((160, 160)),
        dtype=np.float32,
    )

    border = np.concatenate(
        (
            pixels[:24].reshape(-1, 3),
            pixels[-24:].reshape(-1, 3),
            pixels[24:-24, :24].reshape(-1, 3),
            pixels[24:-24, -24:].reshape(-1, 3),
        )
    )

    green = (
        (border[:, 1] > border[:, 0] * 1.12)
        & (border[:, 1] > border[:, 2] * 1.12)
    )

    return {
        "green_edges": float(green.mean()),
        "brightness": float(pixels.mean()),
    }


def detect_objects(image):
    model = object_detector()
    result = model(
        np.asarray(image),
        device="cpu",
        verbose=False,
    )[0]

    boxed = result.plot()[:, :, ::-1].copy()

    objects = []
    people = []

    if result.boxes is not None:
        for box, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
        ):
            label = model.names[int(cls)]
            objects.append(label)

            if label == "person":
                people.append(tuple(map(int, box)))

    return boxed, objects, people


def detect_environment(image, objects):
    labels = set(objects)
    appearance = visual_details(image)

    if labels.intersection(
        {"car", "bus", "truck", "traffic light", "motorcycle"}
    ):
        return (
            "Street or roadside setting is likely from the visible "
            "transport-related objects. Exact location is uncertain."
        )

    if labels.intersection(
        {"bed", "couch", "tv", "dining table", "refrigerator"}
    ):
        return (
            "An indoor setting is likely from the visible household "
            "objects. Exact room or location is uncertain."
        )

    if labels.intersection({"boat", "surfboard"}):
        return (
            "A water-related or coastal setting may be present. "
            "Exact location is uncertain."
        )

    if appearance["green_edges"] > 0.35:
        return (
            "Green surroundings are visible, suggesting vegetation "
            "or an outdoor/nature-related setting. Exact location "
            "cannot be determined reliably."
        )

    return (
        "The exact environment cannot be determined reliably from "
        "the available visual evidence."
    )


def generate_caption(objects, environment):
    if not objects:
        return (
            "An image is visible, but no supported object labels "
            "were detected with high confidence."
        )

    counts = Counter(objects)
    parts = []

    for label, count in counts.most_common(5):
        if count == 1:
            parts.append(f"one {label}")
        else:
            parts.append(f"{count} {label}s")

    if len(parts) == 1:
        subject = parts[0]
    else:
        subject = ", ".join(parts[:-1]) + " and " + parts[-1]

    return (
        f"The image contains {subject}. "
        f"{environment.split('.')[0]}."
    )


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def detect_emotion(image, people):
    if not people:
        return (
            "Not applicable: no person detected. Human facial "
            "expressions are not estimated for animals or scenery."
        )

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    candidates = []

    for x1, y1, x2, y2 in people:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        found = face_detector().detectMultiScale(
            gray[y1:y2, x1:x2],
            scaleFactor=1.15,
            minNeighbors=8,
            minSize=(48, 48),
        )

        for fx, fy, fw, fh in found:
            candidates.append((fx + x1, fy + y1, fw, fh))

    if not candidates:
        return (
            "No clear human face detected; expression was not "
            "estimated."
        )

    unique_faces = []
    for face in sorted(
        candidates,
        key=lambda item: item[2] * item[3],
        reverse=True,
    ):
        if not any(_iou(face, saved) > 0.35 for saved in unique_faces):
            unique_faces.append(face)

    unique_faces.sort(key=lambda item: item[0])

    estimates = []
    net = emotion_net()

    for index, (x, y, w, h) in enumerate(unique_faces[:5], start=1):
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (64, 64))
        blob = crop.astype(np.float32).reshape(1, 1, 64, 64)

        net.setInput(blob)
        scores = net.forward().reshape(-1)

        scores = scores - scores.max()
        probs = np.exp(scores)
        probs = probs / probs.sum()

        best = int(np.argmax(probs))
        label = EMOTION_LABELS[best]
        confidence = float(probs[best]) * 100

        if len(unique_faces) == 1:
            estimates.append(
                f"Estimated facial expression: {label} "
                f"({confidence:.0f}% model confidence)"
            )
        else:
            estimates.append(
                f"Face {index}: {label} "
                f"({confidence:.0f}% model confidence)"
            )

    if not estimates:
        return "Expression estimation was unavailable for this image."

    return "\n".join(estimates)


def generate_description(
    caption,
    objects,
    emotion,
    environment,
    image,
):
    details = visual_details(image)

    if details["brightness"] < 70:
        lighting = "relatively dark"
    elif details["brightness"] > 190:
        lighting = "relatively bright"
    else:
        lighting = "moderately lit"

    object_text = (
        ", ".join(sorted(set(objects)))
        if objects
        else "no supported object labels"
    )

    return (
        f"Main visual: {caption}\n\n"
        f"Detected objects: {object_text}.\n\n"
        f"Setting: {environment}\n\n"
        f"Lighting: The image appears {lighting}.\n\n"
        f"Facial-expression analysis: {emotion}\n\n"
        "Note: facial-expression recognition is an automated visual "
        "estimate and should not be treated as a person's confirmed "
        "emotional state."
    )


def generate_story(caption, objects):
    labels = set(objects)

    if "bird" in labels:
        subject = "bird"
    elif "dog" in labels:
        subject = "dog"
    elif "cat" in labels:
        subject = "cat"
    elif "person" in labels:
        subject = "person"
    else:
        subject = "scene"

    if subject == "bird":
        body = (
            "The bird paused for a moment, watching the surroundings "
            "before moving on. A small sound nearby caught its "
            "attention and turned the quiet pause into a brief "
            "imagined adventure. After exploring the area, it "
            "returned to a safe perch and settled again."
        )
    elif subject == "dog":
        body = (
            "The dog noticed something nearby and became curious. "
            "It explored carefully, stopping now and then to listen "
            "and look around. After a short adventure, it returned "
            "to a familiar spot and relaxed."
        )
    elif subject == "cat":
        body = (
            "The cat watched the surroundings closely as something "
            "caught its attention. Curiosity led it to investigate "
            "for a while. Once satisfied, it returned to a comfortable "
            "place and settled down."
        )
    elif subject == "person":
        body = (
            "The person paused for a photograph during an ordinary "
            "moment. A small detail in the surroundings encouraged "
            "them to stay a little longer and notice the scene more "
            "carefully. The image became a simple reminder of that "
            "brief pause."
        )
    else:
        body = (
            "The visible scene became the beginning of a small "
            "imagined moment. A subtle change in the surroundings "
            "encouraged a closer look, and the ordinary scene became "
            "something worth remembering."
        )

    return (
        "Fictional story inspired by the image:\n\n"
        f"{caption}\n\n{body}"
    )


def encode_image(image_array):
    image = Image.fromarray(
        np.asarray(image_array, dtype=np.uint8)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii")
    )


@app.get("/")
def root():
    return {
        "name": "AI Visual Intelligence Backend",
        "status": "online",
        "frontend": "Vercel",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    data = await file.read()

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB.",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1280, 1280))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image.",
        ) from exc

    try:
        boxed, objects, people = detect_objects(image)
        environment = detect_environment(image, objects)
        caption = generate_caption(objects, environment)

        try:
            emotion = detect_emotion(image, people)
        except Exception:
            emotion = (
                "Facial-expression estimation is temporarily "
                "unavailable."
            )

        description = generate_description(
            caption,
            objects,
            emotion,
            environment,
            image,
        )
        story = generate_story(caption, objects)

        return {
            "boxed_image": encode_image(boxed),
            "caption": caption,
            "objects": sorted(set(objects)),
            "objects_text": (
                "\n".join(
                    f"• {item}"
                    for item in sorted(set(objects))
                )
                or "No supported objects detected."
            ),
            "emotion": emotion,
            "environment": environment,
            "description": description,
            "story": story,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Image analysis failed on the backend. "
                "Please try another image."
            ),
        ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )
