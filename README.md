---
title: Image Captioning with Salesforce BLIP-2
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Image Captioning with Salesforce BLIP-2

The project/folder name is **Image Captioning with Salesforce BLIP-2**. The app heading remains **AI Visual Intelligence System**.

The current main app preserves the uploaded project's BLIP-base caption model, YOLOv8n, FLAN-T5-base and DeepFace. The separate original `app_blip2_backup.py` uses Salesforce BLIP-2 OPT-2.7B for captions, questions and audio. Naming the project does not switch the main model.

## Run locally

Use a fresh Python 3.11 environment. Open a terminal inside this folder:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. The first analysis downloads the required models; the first detected human face also triggers emotion-model loading. Allow several GB of disk/RAM and internet for downloads. CPU analysis may be slow.

## Changed files

- `app.py`: preserves the original seven-output Gradio layout. Models load on first use. OpenCV detector data is resolved and checked. Only detected face crops reach the expression model; no face and model failure produce different messages. Up to five faces are shown. Image colours are preserved with BGR/RGB conversion. Scene summary uses caption/object estimates; story uses those directly rather than an invented intermediate description. A failed optional stage leaves other results visible. Hosting address and port can be set with environment variables.
- `requirements.txt`: adds missing NumPy, YOLO, DeepFace, OpenCV and TensorFlow/Keras dependencies with bounded versions. Target Python 3.11. This is a proposed configuration, not a clean-install-verified lockfile.
- `app_blip2_backup.py`: unchanged original backup; `requirements-backup.txt` adds its missing gTTS dependency.
- `Dockerfile`, `.dockerignore`, `.gitignore`: persistent Python hosting configuration and exclusions for model weights, secrets and environments.
- `tests/test_app.py`: targeted checks using mocked model inference.

The uploaded app already converted YOLO BGR to RGB and enforced face detection. The screenshots with swapped colours and animal/landscape emotion labels may therefore show a different revision. Haar face detection can still miss faces or produce false positives.

## Checks and limitations

Run `python -m unittest discover -s tests -v`.

Mocked checks are not proof of real caption/expression/story accuracy. Full inference, clean installation, Docker build and hosted deployment remain unverified. The required BLIP/FLAN-T5 models and OpenCV/DeepFace were not available in the earlier test environment. FLAN-T5-base may still produce short or poor stories; the revised prompt does not guarantee fluent output. Scene description is now a concise summary of model estimates instead of speculative paragraphs.

Before publishing, test a parrot, a landscape, a human face, multiple faces and a low-light photo in a clean environment. Verify colours, captions and story relevance. Save a dependency lock after a successful installation.

## Hosting and GitHub

The Dockerfile targets a persistent Python service such as a Hugging Face Docker Space, using port 7860. See https://huggingface.co/docs/hub/spaces-sdks-docker.

The main app starts a Gradio server; it is not configured as a Vercel serverless entry point. A separate Vercel frontend could later link to or embed the hosted app. No live backend, GitHub push or Vercel deployment was created. See https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size for current large-function support; a larger bundle limit alone does not establish compatibility.

Copy these source files into your existing repository after review. Keep model weights, virtual environments, generated audio and credentials out of Git. Do not overwrite unrelated files.

## Archived backup findings

The BLIP-2 backup loads a much larger model at import, lacks explicit inference-mode guards and model.eval(), does not handle TTS network errors, and leaves temporary audio files behind. Its MPS half-precision path needs hardware testing. It remains unchanged to preserve the original experiment; it is not the emotion/story entry point.
