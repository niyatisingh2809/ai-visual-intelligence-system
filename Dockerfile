FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 10000

CMD ["python", "app.py"]
