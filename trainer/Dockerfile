FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# This trainer has no GUI. Avoid apt-based X11/XCB dependencies entirely.
# Ultralytics declares opencv-python as a dependency; replace it with the
# headless build, which exposes the same cv2 API without desktop Qt/X11.
RUN pip install --no-cache-dir ultralytics==8.4.171 \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless \
    && pip install --no-cache-dir opencv-python-headless==4.12.0.88

# Make a missing shared library fail during the build, not on a GPU job.
RUN python -c "import cv2; from ultralytics import YOLO; print('OpenCV OK:', cv2.__version__)"

WORKDIR /work
# Populate approved checkpoints during build; runtime containers have no network access.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt'); YOLO('yolo26s.pt')"

COPY trainer/train.py /work/train.py
ENTRYPOINT ["python", "/work/train.py"]
