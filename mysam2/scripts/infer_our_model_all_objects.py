import os
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# ====================== SAM2 ======================
from sam2.build_sam import build_sam2_video_predictor

# ====================== CONFIG ======================
MOT17_SEQUENCE = "MOT17-02-FRCNN"

BASE_PATH = f"/home/vishnu/Ajitesh/mtp/datasets/processed/mot17/train/{MOT17_SEQUENCE}"
VIDEO_FOLDER = f"{BASE_PATH}/img1"
DET_PATH = f"{BASE_PATH}/det/det.txt"

OUTPUT_FOLDER = f"results/{MOT17_SEQUENCE}_SAM2_DET_PROMPT_ALL"

CONFIG_NAME = "configs/samurai/sam2.1_hiera_s.yaml"
CKPT_PATH = "/home/vishnu/Ajitesh/mtp/phase2/final_model/sam2/checkpoints/sam2.1_hiera_small.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== MODES ======================
DAM_MODE = True
SAMURAI_MODE = True

MEMORY_UPDATE_STRIDE = 5
ANCHOR_THRESHOLD = 0.7
IOU_THRESHOLD = 0.8
AREA_THRESHOLD = 0.2
MEDIAN_WINDOW = 10

STABLE_FRAMES_THRESHOLD = 15
STABLE_IOUS_THRESHOLD = 0.3

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ====================== LOAD DETECTIONS ======================
def load_mot_detections(det_path, conf_thresh=0.5):
    dets = {}
    with open(det_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame = int(parts[0]) # MOT frames are 1-indexed
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            if conf < conf_thresh:
                continue
            bbox = [x, y, x + w, y + h] # xyxy in original image space
            dets.setdefault(frame, []).append(bbox)
    return dets

print("Loading detections...")
detections = load_mot_detections(DET_PATH)

# ====================== LOAD MODEL ======================
print("Loading SAM 2.1 with DAM4SAM + SAMURAI...")
predictor = build_sam2_video_predictor(
    config_file=CONFIG_NAME,
    ckpt_path=CKPT_PATH,
    device=DEVICE,
    dam_mode=DAM_MODE,
    memory_update_stride=MEMORY_UPDATE_STRIDE,
    anchor_threshold=ANCHOR_THRESHOLD,
    iou_threshold=IOU_THRESHOLD,
    area_threshold=AREA_THRESHOLD,
    median_window=MEDIAN_WINDOW,
    samurai_mode=SAMURAI_MODE,
    stable_frames_threshold=STABLE_FRAMES_THRESHOLD,
    stable_ious_threshold=STABLE_IOUS_THRESHOLD,
    memory_bank_iou_threshold=0.7, # τm
    memory_bank_obj_score_threshold=0.5, # τo
    memory_bank_kf_score_threshold=0.3, # τk
    kf_score_weight=0.2, # α in paper
)
print("Model loaded successfully!")

# ====================== CREATE INFERENCE STATE ======================
print(f"\nLoading frames from: {VIDEO_FOLDER}")
inference_state = predictor.init_state(
    video_path=VIDEO_FOLDER, # SAM2 will read all jpg/png in the folder
    offload_video_to_cpu=True,
    offload_state_to_cpu=True,
)

# optional: limit to first 300 frames like your original script
MAX_FRAMES = 300
if inference_state["num_frames"] > MAX_FRAMES:
    inference_state["images"] = inference_state["images"][:MAX_FRAMES]
    inference_state["num_frames"] = MAX_FRAMES
    # clear any cached features beyond the limit
    inference_state["cached_features"] = {
        k: v for k, v in inference_state["cached_features"].items() if k < MAX_FRAMES
    }

print(f"Loaded {inference_state['num_frames']} frames "
      f"({inference_state['video_height']}x{inference_state['video_width']})")

# ====================== ADD DET PROMPTS (ALL OBJECTS) ======================
print("Adding detection prompts (frame 0) - ALL objects...")
frame0_dets = detections.get(1, []) # frame 1 in MOT = frame_idx 0

# Sort by size (descending) for visualization, but KEEP ALL
frame0_dets = sorted(
    frame0_dets,
    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
    reverse=True
)
# NO slicing [:5] - we keep ALL detections

for obj_id, bbox in enumerate(frame0_dets, start=1):
    # bbox is already in original image coordinates, which matches video_width/height
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        box=bbox,
    )

print(f"Initialized {len(frame0_dets)} objects (ALL detections)")

# ====================== PROPAGATION ======================
print("Running tracking...")

for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
    inference_state,
    start_frame_idx=0,
    max_frame_num_to_track=inference_state["num_frames"],
    reverse=False,
):
    for i, obj_id in enumerate(obj_ids):
        mask = (video_res_masks[i, 0] > 0).cpu().numpy().astype(np.uint8) * 255

        out_dir = Path(OUTPUT_FOLDER) / f"obj_{obj_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{frame_idx:06d}.png"
        Image.fromarray(mask).save(out_path)

    if frame_idx % 50 == 0:
        print(f"Processed frame {frame_idx}")

print("\n✅ DONE!")
print(f"Saved results in: {OUTPUT_FOLDER}")
print(f"Total objects tracked: {len(frame0_dets)}")
