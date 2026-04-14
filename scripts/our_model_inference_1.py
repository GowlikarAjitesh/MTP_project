import torch
import cv2
import numpy as np
from pathlib import Path
from sam2.sam2_video_predictor import SAM2VideoPredictor

# ========================== GPU CONFIG ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# ========================== MODEL CONFIG ==========================
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"      # change to "sam2.1_hiera_t.yaml" for faster speed
CHECKPOINT = "checkpoints/sam2.1_hiera_b+.pt"

print("Loading SAM2VideoPredictor with SAM2MOT Trajectory Manager...")
predictor = SAM2VideoPredictor.from_pretrained(
    model_cfg=MODEL_CFG,
    checkpoint=CHECKPOINT,
)

# Force model to GPU
predictor.to(DEVICE)          # ← Important for full GPU usage

# ========================== VIDEO ==========================
VIDEO_PATH = "path/to/your/short/video.mp4"   # ← CHANGE THIS
# VIDEO_PATH = 0                                  # ← uncomment for webcam

inference_state = predictor.init_state(
    video_path=VIDEO_PATH,
    offload_video_to_cpu=False,      # keep on GPU for speed
    offload_state_to_cpu=False,
)

print("✅ Ready! Click on objects to start tracking. Press 'q' to quit.")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run one frame with full SAM2MOT manager
    for out_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=frame_idx,
        max_frame_num_to_track=1,
        detections=None,                    # None = pure interactive mode
    ):
        vis = frame.copy()

        for i, mask_tensor in enumerate(video_res_masks):
            mask = mask_tensor.cpu().numpy() > 0
            color = np.array([0, 255, 0], dtype=np.uint8)
            vis[mask] = vis[mask] * 0.5 + color * 0.5

            if len(obj_ids) > i:
                y, x = np.nonzero(mask)
                if len(y) > 0:
                    cv2.putText(vis, f"ID{obj_ids[i]}", (x.min(), y.min() - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("SAM2 + SAM2MOT (GPU)", vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):   # press C to reset all tracks
        predictor.reset_state(inference_state)
        print("All tracks reset")

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()
print("✅ Test finished!")