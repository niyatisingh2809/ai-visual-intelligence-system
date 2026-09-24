import os
import logging
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cpu"


@lru_cache(maxsize=1)
def caption_components():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name).to(device).eval()
    return processor, model


@lru_cache(maxsize=1)
def text_components():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def object_detector():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


@torch.inference_mode()
def generate_caption(image):
    processor, model = caption_components()
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
    )
    return processor.decode(
        output[0],
        skip_special_tokens=True,
    ).strip()


def detect_objects(image):
    model = object_detector()
    result = model(image, device=device, verbose=False)[0]

    boxed_image = result.plot()[:, :, ::-1].copy()

    objects = sorted(
        {
            model.names[int(cls)]
            for cls in result.boxes.cls
        }
    )

    people = [
        tuple(map(int, box))
        for box, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.tolist(),
        )
        if model.names[int(cls)] == "person"
    ]

    formatted = (
        "\n".join(f"• {obj}" for obj in objects)
        or "No objects detected."
    )

    return boxed_image, objects, formatted, people


@lru_cache(maxsize=1)
def face_detector():
    import cv2

    path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )

    if not path.is_file():
        raise RuntimeError(
            f"OpenCV face detector data missing: {path}"
        )

    detector = cv2.CascadeClassifier(str(path))

    if detector.empty():
        raise RuntimeError(
            f"OpenCV face detector could not load: {path}"
        )

    return detector


def _iou(face_a, face_b):
    ax, ay, aw, ah = face_a
    bx, by, bw, bh = face_b

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection

    return intersection / union if union else 0.0


def detect_emotion(image, people=None):
    if people is None:
        return (
            "Human detection unavailable; "
            "facial expression was not estimated."
        )

    if not people:
        return (
            "Not applicable: no person detected. "
            "Human facial expressions cannot be estimated "
            "for animals or scenery."
        )

    try:
        import cv2

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

            roi_gray = gray[y1:y2, x1:x2]

            found = face_detector().detectMultiScale(
                roi_gray,
                scaleFactor=1.15,
                minNeighbors=10,
                minSize=(50, 50),
            )

            if len(found) == 0:
                continue

            # Keep the strongest/largest face candidate for
            # each independently detected person.
            fx, fy, fw, fh = max(
                found,
                key=lambda face: face[2] * face[3],
            )

            candidates.append(
                (fx + x1, fy + y1, fw, fh)
            )

        if not candidates:
            return (
                "No clear human face detected; "
                "expression not estimated."
            )

        # Remove overlapping duplicates.
        unique_faces = []

        for face in sorted(
            candidates,
            key=lambda item: item[2] * item[3],
            reverse=True,
        ):
            if not any(
                _iou(face, saved) > 0.35
                for saved in unique_faces
            ):
                unique_faces.append(face)

        unique_faces.sort(key=lambda item: item[0])

        from deepface import DeepFace

        estimates = []

        for index, (x, y, w, h) in enumerate(
            unique_faces[:5],
            start=1,
        ):
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)

            sx = max(0, x - pad_x)
            sy = max(0, y - pad_y)
            ex = min(rgb.shape[1], x + w + pad_x)
            ey = min(rgb.shape[0], y + h + pad_y)

            face_crop = rgb[sy:ey, sx:ex].copy()

            result = DeepFace.analyze(
                img_path=face_crop,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                silent=True,
            )

            item = (
                result[0]
                if isinstance(result, list)
                else result
            )

            emotion = str(
                item.get("dominant_emotion", "unknown")
            ).lower()

            if len(unique_faces) == 1:
                estimates.append(
                    f"Detected facial expression: {emotion}"
                )
            else:
                estimates.append(
                    f"Face {index} (left to right): {emotion}"
                )

        if len(unique_faces) > 5:
            estimates.append(
                "Showing the first 5 detected faces."
            )

        return "\n".join(estimates)

    except (ImportError, RuntimeError):
        logger.exception(
            "Expression dependencies or detector unavailable"
        )
        return (
            "Expression analysis unavailable: check installed "
            "dependencies and OpenCV detector data."
        )

    except Exception:
        logger.exception("Expression inference failed")
        return (
            "Expression analysis unavailable: model loading "
            "or inference failed; check server logs."
        )


def visual_details(image):
    """Describe simple visible properties without inventing a location."""

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


def detect_environment(caption, image=None):
    locations = {
        "Forest / woodland": (
            "forest",
            "woods",
            "woodland",
        ),
        "Park": ("park",),
        "Street / roadside": (
            "street",
            "road",
            "sidewalk",
        ),
        "Indoor room": (
            "room",
            "kitchen",
            "bedroom",
            "indoors",
        ),
        "Beach / coast": (
            "beach",
            "seashore",
        ),
        "Office": ("office",),
        "Mountain landscape": (
            "mountain",
            "mountains",
        ),
        "Field / meadow": (
            "field",
            "meadow",
        ),
    }

    caption_lower = caption.lower()

    matches = [
        label
        for label, words in locations.items()
        if any(
            re.search(
                r"\b" + re.escape(word) + r"\b",
                caption_lower,
            )
            for word in words
        )
    ]

    if matches:
        return (
            " + ".join(matches)
            + " (estimated from image caption)."
        )

    nature_words = (
        r"\b(branch|tree|leaves|grass|flower|flowers|"
        r"plant|plants|bird|parrot)\b"
    )

    if re.search(nature_words, caption_lower):
        return (
            "Nature-related visual clues are present. "
            "The exact location cannot be determined "
            "reliably from the image alone."
        )

    if image is not None:
        appearance = visual_details(image)

        if appearance["green_edges"] > 0.35:
            return (
                "Green surroundings are visible, possibly "
                "vegetation or decorative greenery. "
                "The exact location is uncertain."
            )

    return (
        "The main subject is visible, but there are not "
        "enough reliable clues to determine the exact setting."
    )


def generate_scene_description(
    caption,
    objects,
    emotion,
    environment,
    image=None,
):
    opening = (
        caption[:1].upper()
        + caption[1:].rstrip(". ")
        + "."
    )

    parts = [
        f"Main visual: {opening}"
    ]

    if objects:
        parts.append(
            "Detected objects: "
            + ", ".join(objects)
            + "."
        )
    else:
        parts.append(
            "No additional object labels were detected."
        )

    parts.append(
        "Setting: " + environment
    )

    if image is not None:
        appearance = visual_details(image)

        if appearance["green_edges"] > 0.35:
            parts.append(
                "Visible appearance: green tones occupy a "
                "noticeable part of the image background."
            )

        if appearance["brightness"] < 70:
            parts.append(
                "Lighting: the image appears relatively dark."
            )
        elif appearance["brightness"] > 190:
            parts.append(
                "Lighting: the image appears relatively bright."
            )
        else:
            parts.append(
                "Lighting: the image has moderate overall brightness."
            )

    parts.append(
        "Facial-expression analysis: " + emotion
    )

    parts.append(
        "Note: facial-expression recognition is only an "
        "automated visual estimate and should not be treated "
        "as a person's confirmed emotional state."
    )

    return "\n\n".join(parts)


def _story_subject(caption, objects):
    lower = caption.lower()

    if (
        "bird" in objects
        or re.search(
            r"\b(parrot|bird|sparrow|eagle|owl)\b",
            lower,
        )
    ):
        return "bird"

    if "person" in objects:
        if re.search(r"\b(woman|girl|lady)\b", lower):
            return "woman"
        if re.search(r"\b(man|boy|gentleman)\b", lower):
            return "man"
        return "person"

    if "dog" in objects or "dog" in lower:
        return "dog"

    if "cat" in objects or "cat" in lower:
        return "cat"

    return "scene"


def fallback_story(caption, objects):
    opening = (
        caption[:1].upper()
        + caption[1:].rstrip(". ")
        + "."
    )

    subject = _story_subject(caption, objects)

    if subject == "bird":
        paragraphs = [
            (
                f"{opening} In this fictional story, the bird "
                "had paused on its perch while listening to "
                "the sounds around it. A distant call caught "
                "its attention, and curiosity soon turned the "
                "quiet moment into the beginning of a journey."
            ),
            (
                "The bird moved from branch to branch, following "
                "the call while the surroundings changed around "
                "it. For a moment the sound disappeared, so it "
                "stopped instead of flying farther and listened "
                "carefully for another clue."
            ),
            (
                "Soon the familiar call returned from nearby. "
                "The bird followed it to a sheltered perch and "
                "settled there safely. What began as an ordinary "
                "pause had become a small imagined adventure "
                "inspired by the bird visible in the image."
            ),
        ]

    elif subject in {"woman", "man", "person"}:
        if subject == "woman":
            character = "the woman"
        elif subject == "man":
            character = "the man"
        else:
            character = "the person"

        paragraphs = [
            (
                f"{opening} In this fictional story, {character} "
                "had stopped for a photograph before continuing "
                "with the day. The quiet portrait captured a "
                "brief moment that might otherwise have passed "
                "without much notice."
            ),
            (
                f"After the photograph, {character} noticed a "
                "small detail in the surroundings that had been "
                "easy to overlook. That observation became a "
                "reason to stay for a few more minutes and enjoy "
                "the setting rather than immediately moving on."
            ),
            (
                f"By the time {character} finally left, the image "
                "had become a reminder of that simple pause. "
                "Nothing dramatic needed to happen; the fictional "
                "story grew from the subject and visible scene "
                "captured in the photograph."
            ),
        ]

    elif subject == "dog":
        paragraphs = [
            (
                f"{opening} In this fictional story, the dog "
                "noticed an unfamiliar sound nearby and became "
                "curious about where it came from."
            ),
            (
                "It explored the area carefully, stopping now "
                "and then to listen and look around before "
                "continuing."
            ),
            (
                "Eventually the dog returned to its familiar "
                "spot, satisfied with the small adventure and "
                "ready to rest."
            ),
        ]

    elif subject == "cat":
        paragraphs = [
            (
                f"{opening} In this fictional story, the cat "
                "watched the surroundings carefully as something "
                "nearby caught its attention."
            ),
            (
                "Its curiosity led it to investigate, but it "
                "moved slowly and cautiously through the scene."
            ),
            (
                "After deciding there was nothing to worry about, "
                "the cat returned to a comfortable place and "
                "settled down again."
            ),
        ]

    else:
        paragraphs = [
            (
                f"{opening} In this fictional story, the visible "
                "scene became the beginning of a small moment "
                "worth remembering."
            ),
            (
                "A subtle change in the surroundings created a "
                "brief interruption and encouraged the observer "
                "to pay closer attention to the details."
            ),
            (
                "The moment soon passed, but the image remained "
                "as a reminder that even an ordinary scene can "
                "inspire a simple story."
            ),
        ]

    return (
        "Fictional story inspired by the image:\n\n"
        + "\n\n".join(paragraphs)
    )


def story_is_grounded(story, caption, objects):
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        story.lower(),
    )

    if len(words) < 65:
        return False

    anchor_text = " ".join(
        [caption] + list(objects)
    ).lower()

    anchors = {
        word
        for word in re.findall(
            r"\b[a-zA-Z]{4,}\b",
            anchor_text,
        )
        if word not in {
            "with",
            "that",
            "this",
            "there",
            "image",
            "sitting",
            "standing",
        }
    }

    if not anchors:
        return True

    return bool(
        anchors.intersection(set(words))
    )


@torch.inference_mode()
def generate_story(caption, objects):
    """
    Try a lightweight text model first.
    If its result is short, repetitive, or poorly grounded,
    use a deterministic image-grounded fallback.
    """

    try:
        tokenizer, model = text_components()

        prompt = (
            "Write a short fictional story inspired ONLY by the "
            "following image information. Do not introduce sports, "
            "holidays, locations, objects, or people that are not "
            "supported by the image information. Write exactly "
            "three coherent paragraphs with a beginning, a small "
            "development, and an ending.\n\n"
            f"Image caption: {caption}\n"
            f"Detected objects: {', '.join(objects) if objects else 'none'}\n\n"
            "Story:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        story = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        if story_is_grounded(
            story,
            caption,
            objects,
        ):
            return (
                "Fictional story inspired by the image:\n\n"
                + story
            )

        logger.info(
            "Generated story was not sufficiently grounded; "
            "using fallback."
        )

    except Exception:
        logger.exception(
            "Story model unavailable; using grounded fallback."
        )

    return fallback_story(
        caption,
        objects,
    )


def analyze_image(image):
    if image is None:
        raise gr.Error(
            "Please upload an image first."
        )

    image = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    image.thumbnail(
        (1600, 1600)
    )

    try:
        caption = generate_caption(image)
    except Exception:
        logger.exception(
            "Caption model failed"
        )
        raise gr.Error(
            "Caption model could not load or run. "
            "Check model downloads and server logs."
        )

    try:
        (
            boxed_image,
            objects,
            formatted,
            people,
        ) = detect_objects(image)

    except Exception:
        logger.exception(
            "Object detection failed"
        )

        boxed_image = np.asarray(image)
        objects = []
        formatted = (
            "Object detection unavailable; "
            "check server logs."
        )
        people = None

    emotion = detect_emotion(
        image,
        people,
    )

    environment = detect_environment(
        caption,
        image,
    )

    description = generate_scene_description(
        caption,
        objects,
        emotion,
        environment,
        image,
    )

    try:
        story = generate_story(
            caption,
            objects,
        )
    except Exception:
        logger.exception(
            "Story generation failed"
        )
        story = fallback_story(
            caption,
            objects,
        )

    return (
        boxed_image,
        caption,
        formatted,
        emotion,
        environment,
        description,
        story,
    )


CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

#hero {
    text-align: center;
    padding: 18px 10px 8px 10px;
}

#hero h1 {
    margin-bottom: 6px;
}

#analyze-button {
    min-height: 48px;
    font-weight: 700;
}

.output-card textarea {
    font-size: 15px !important;
    line-height: 1.55 !important;
}
"""


with gr.Blocks(
    title="AI Visual Intelligence System",
    css=CSS,
) as demo:

    gr.Markdown(
        """
<div id="hero">

# 🎨 AI Visual Intelligence System

### Image Understanding • Object Detection • Scene Analysis • Creative Storytelling

Upload an image and let AI generate a caption, detect visible objects,
estimate facial expressions when a clear human face is available,
describe the scene, and create a fictional story inspired by the image.

</div>
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="📸 Upload Image",
            height=400,
        )

        boxed_image_output = gr.Image(
            label="🔍 Objects Detected",
            height=400,
        )

    analyze_btn = gr.Button(
        "🚀 Analyze Image",
        variant="primary",
        size="lg",
        elem_id="analyze-button",
    )

    with gr.Row():
        caption_output = gr.Textbox(
            label="📝 Auto Caption",
            lines=5,
            max_lines=14,
            min_width=300,
            interactive=False,
            elem_classes=["output-card"],
        )

        emotion_output = gr.Textbox(
            label="😊 Facial Expression Estimate",
            info=(
                "Automated facial-expression estimate; "
                "not a confirmed feeling."
            ),
            lines=5,
            max_lines=14,
            min_width=300,
            interactive=False,
            elem_classes=["output-card"],
        )

        environment_output = gr.Textbox(
            label="🏞️ Environment",
            lines=5,
            max_lines=14,
            min_width=300,
            interactive=False,
            elem_classes=["output-card"],
        )

    objects_output = gr.Textbox(
        label="📋 Objects Detected",
        max_lines=5,
        interactive=False,
        elem_classes=["output-card"],
    )

    description_output = gr.Textbox(
        label="🌟 Deep Scene Description",
        lines=12,
        max_lines=30,
        interactive=False,
        show_copy_button=True,
        elem_classes=["output-card"],
    )

    with gr.Accordion(
        "📖 Generated Story",
        open=True,
    ):
        story_output = gr.Textbox(
            label="✍️ Fiction Inspired by the Image",
            lines=15,
            max_lines=40,
            interactive=False,
            show_copy_button=True,
            elem_classes=["output-card"],
        )

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[
            boxed_image_output,
            caption_output,
            objects_output,
            emotion_output,
            environment_output,
            description_output,
            story_output,
        ],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        show_error=False,
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "127.0.0.1",
        ),
        server_port=int(
            os.getenv(
                "PORT",
                os.getenv(
                    "GRADIO_SERVER_PORT",
                    "7860",
                ),
            )
        ),
    )
