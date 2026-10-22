FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# OpenCV dynamic libraries required by Ultralytics in the minimal CUDA base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ultralytics==8.4.171
RUN python -c "import cv2; print('OpenCV:', cv2.__version__)"
WORKDIR /work
# Populate approved checkpoints during build; runtime containers have no network access.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26s.pt')"
COPY trainer/train.py /work/train.py
ENTRYPOINT ["python", "/work/train.py"]
