import cv2
import os
import warnings
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================
# CONFIG
# ==============================

VIDEO_PATH = "/home/cs24m118/datasets/videos/sam_test_pegion.mp4"
OUTPUT_PATH = "/home/cs24m118/phase2/deepSort/outputoutput_deepSort.avi"         # AVI is safer on servers
CONF_THRESH = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# SANITY CHECK: INPUT VIDEO
# ==============================
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Failed to open input video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if width == 0 or height == 0:
    raise RuntimeError("Invalid video dimensions")

if fps == 0 or fps is None:
    fps = 25.0  # fallback FPS

print(f"✅ Video loaded: {width}x{height} @ {fps:.2f} FPS")

# ==============================
# VIDEO WRITER (SAFE)
# ==============================
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    raise RuntimeError("Failed to open VideoWriter")

print("✅ VideoWriter opened")

# ==============================
# LOAD MODELS
# ==============================
print("⏳ Loading YOLOv8...")
model = YOLO("yolov8n.pt").to(DEVICE)

print("⏳ Initializing DeepSORT...")
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=100
)

# ==============================
# MAIN LOOP
# ==============================
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # YOLO detection
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            detections.append(([x1, y1, w, h], conf, cls_id))

    # DeepSORT update
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (l, max(0, t - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)

    if frame_id % 50 == 0:
        print(f"✅ Processed frame {frame_id}")

# ==============================
# CLEANUP
# ==============================
cap.release()
out.release()

print(f"🎉 DONE. Output saved to: {OUTPUT_PATH}")
print(f"📊 Total frames processed: {frame_id}")
