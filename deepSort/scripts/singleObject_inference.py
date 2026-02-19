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
OUTPUT_PATH = "/home/cs24m118/phase2/deepSort/output/output_single_object.avi"
CONF_THRESH = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# SANITY CHECK
# ==============================
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Failed to open input video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("Failed to open VideoWriter")

print("✅ Video and writer initialized")

# ==============================
# LOAD MODELS
# ==============================
model = YOLO("yolov8n.pt").to(DEVICE)
tracker = DeepSort(max_age=30, n_init=3)

# ==============================
# SINGLE OBJECT CONTROL
# ==============================
target_track_id = 6
target_initialized = True

# ==============================
# MAIN LOOP
# ==============================
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

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

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        # 🔒 Lock onto FIRST object
        if not target_initialized:
            target_track_id = track.track_id
            target_initialized = True
            print(f"🎯 Locked on target ID: {target_track_id}")

        # Ignore all other objects
        if track.track_id != target_track_id:
            continue

        l, t, w, h = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"TARGET ID {track.track_id}",
            (l, max(0, t - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    out.write(frame)

    if frame_id % 50 == 0:
        print(f"Processed frame {frame_id}")

# ==============================
# CLEANUP
# ==============================
cap.release()
out.release()

print("🎉 DONE. Single-object tracking completed.")
print(f"📁 Output: {OUTPUT_PATH}")
