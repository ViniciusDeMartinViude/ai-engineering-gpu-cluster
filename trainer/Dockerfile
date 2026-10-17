FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN pip install --no-cache-dir ultralytics==8.4.171
WORKDIR /work
# Populate approved checkpoints during build; runtime containers have no network access.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26s.pt')"
COPY trainer/train.py /work/train.py
ENTRYPOINT ["python", "/work/train.py"]
