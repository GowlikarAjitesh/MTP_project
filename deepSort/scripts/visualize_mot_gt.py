import cv2
import os
import numpy as np
from collections import defaultdict

VIDEO_PATH = "/home/cs24m118/sav_dataset/sav_train/sav_000/sav_000001.mp4"
GT_PATH = "/home/cs24m118/sav_dataset/gt_mot/sav_000001/gt.txt"
OUT_VIDEO = "/home/cs24m118/phase2/deepSort/output/gt_mot/sav_000001/gt_vis.mp4"

def load_mot_gt(gt_path):
    gt = defaultdict(list)
    if not os.path.exists(gt_path):
        print(f"⚠️ Warning: GT file not found at {gt_path}")
        return gt
        
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6: continue
            frame, tid, x, y, w, h = map(float, parts[:6])
            gt[int(frame)].append((int(tid), int(x), int(y), int(w), int(h)))
    return gt

def get_color(idx):
    """Generates a pseudo-random color for a given ID."""
    np.random.seed(idx)
    return tuple(map(int, np.random.randint(0, 255, 3)))

def main():
    # Ensure output path exists
    os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

    gt = load_mot_gt(GT_PATH)
    frame_idx = 1 

    print("🎞️  Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_idx in gt:
            for tid, x, y, w, h in gt[frame_idx]:
                color = get_color(tid)
                # Drawing logic
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID {tid}", (x, max(15, y - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"✅ Saved visualization: {OUT_VIDEO}")

if __name__ == "__main__":
    main()