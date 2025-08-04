FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] cellpose opencv-python-headless numpy Pillow runpod

# Preload model weights
RUN python3 -c "from cellpose import models; models.CellposeModel(gpu=True)"

ENV HOME=/root

CMD ["python3", "-u", "rp_handler.py"]
