"""
SAM 2 Multi-Object Tracker for MOT17 - Memory Optimized Version
"""

import argparse
import collections
import configparser
import json
import os
import torch
import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

# SAM 2 imports
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# ====================== MEMORY OPTIMIZATIONS ======================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

mm.lap.default_solver = "scipy"

PEDESTRIAN_CLASS_ID = 1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: np.ndarray   # [x, y, w, h]
    score: float


@dataclass
class Track:
    obj_id: int
    last_bbox: np.ndarray


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 2 MOT17 Tracker - Memory Optimized")
    parser.add_argument("--mot-root", default="/home/cs24m118/datasets/MOT17/train")
    parser.add_argument("--output-root", default="/home/cs24m118/phase2/outputs/sam2_tracking")
    parser.add_argument("--seq", nargs="*", help="Specific sequences to run")
    parser.add_argument("--det-score-thresh", type=float, default=0.6)   # Slightly higher for stability

    # Use base_plus model by default (strongly recommended)
    parser.add_argument(
        "--sam2-checkpoint",
        default="/home/cs24m118/phase2/mysam2/checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to SAM 2 checkpoint (use base_plus to avoid OOM)"
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",   # config name can stay, model size is in checkpoint
        help="SAM 2 config"
    )
    parser.add_argument("--eval-iou-thresh", type=float, default=0.5)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers (unchanged)
# ---------------------------------------------------------------------------

def load_seqinfo(seq_path):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, "seqinfo.ini"))
    s = seqinfo["Sequence"]
    return {
        "name": s["name"], "im_dir": s["imDir"], "frame_rate": int(s["frameRate"]),
        "seq_length": int(s["seqLength"]), "width": int(s["imWidth"]),
        "height": int(s["imHeight"]), "ext": s["imExt"]
    }


def load_detections(det_path, score_thresh):
    detections_by_frame = {}
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7: continue
            frame_id = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6])
            if score < score_thresh: continue
            detections_by_frame.setdefault(frame_id, []).append(
                Detection(bbox=np.array([x, y, w, h], dtype=np.float32), score=score)
            )
    return detections_by_frame


def load_ground_truth(gt_path):
    gt_by_frame = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9: continue
            frame_id = int(parts[0])
            gt_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = int(float(parts[6]))
            cls = int(float(parts[7]))
            if conf != 1 or cls != PEDESTRIAN_CLASS_ID: continue
            gt_by_frame.setdefault(frame_id, []).append(
                {"id": gt_id, "bbox": np.array([x, y, w, h], dtype=np.float32)}
            )
    return gt_by_frame


# ---------------------------------------------------------------------------
# Geometry helpers (unchanged)
# ---------------------------------------------------------------------------

def mask_to_bbox(mask: np.ndarray):
    if mask.sum() == 0: return None
    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1], dtype=np.float32)


def bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0.0
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main tracking function
# ---------------------------------------------------------------------------

def run_tracker_on_sequence(seq_name, seq_path, args, results_dir, videos_dir):
    seqinfo = load_seqinfo(seq_path)
    detections = load_detections(os.path.join(seq_path, "det", "det.txt"), args.det_score_thresh)
    gt_by_frame = load_ground_truth(os.path.join(seq_path, "gt", "gt.txt"))
    image_dir = os.path.join(seq_path, seqinfo["im_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  → Using device: {device} | Model: {os.path.basename(args.sam2_checkpoint)}")

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        args.sam2_config, args.sam2_checkpoint, device=device
    )

    # Important memory saving flags
    inference_state = predictor.init_state(
        video_path=image_dir,
        offload_video_to_cpu=True,      # Major VRAM saver
    )

    next_obj_id = 1
    active_tracks: list[Track] = []
    result_rows = []
    results_by_frame = collections.defaultdict(list)
    tracked_detector_ids: set = set()   # (frame_id, rounded_bbox_tuple)

    print(f"  → Tracking {seq_name} ({seqinfo['seq_length']} frames)")

    for frame_idx in range(seqinfo["seq_length"]):
        frame_id = frame_idx + 1
        frame_dets = detections.get(frame_id, [])

        new_objects_added = 0

        # Add new objects only if not seen before
        for det in frame_dets:
            det_key = (frame_id, tuple(np.round(det.bbox, decimals=1)))
            if det_key in tracked_detector_ids:
                continue

            x, y, w, h = det.bbox
            box = np.array([x, y, x + w, y + h], dtype=np.float32)

            try:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=next_obj_id,
                        box=box,
                    )

                active_tracks.append(Track(obj_id=next_obj_id, last_bbox=det.bbox.copy()))
                tracked_detector_ids.add(det_key)
                next_obj_id += 1
                new_objects_added += 1

            except Exception as e:
                print(f"    Warning: Failed to add object at frame {frame_id}: {e}")
                torch.cuda.empty_cache()

        # Propagate one frame
        frame_masks = {}
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for out_fidx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_idx,
                    max_frame_num_to_track=1,
                ):
                    if out_fidx != frame_idx: continue
                    for i, oid in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                        frame_masks[oid] = mask
        except Exception as e:
            print(f"    Warning: propagate error at frame {frame_id}: {e}")

        # Update tracks
        for trk in active_tracks:
            mask = frame_masks.get(trk.obj_id)
            if mask is not None:
                new_bbox = mask_to_bbox(mask)
                if new_bbox is not None:
                    trk.last_bbox = new_bbox

            x, y, w, h = trk.last_bbox
            row = [frame_id, trk.obj_id,
                   round(float(x), 2), round(float(y), 2),
                   round(float(w), 2), round(float(h), 2),
                   1.0, -1, -1, -1]
            result_rows.append(row)
            results_by_frame[frame_id].append(row)

        torch.cuda.empty_cache()   # Clean after every frame

        if frame_idx % 50 == 0 or frame_idx == 0:
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"    Frame {frame_id}/{seqinfo['seq_length']} | "
                  f"Active: {len(active_tracks)} | New: {new_objects_added} | "
                  f"GPU Mem: {allocated:.0f} MB")

    # --------------------- Save Results ---------------------
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{seq_name}.txt")
    with open(result_path, "w") as f:
        for row in result_rows:
            f.write(",".join(map(str, row)) + "\n")

    # --------------------- Video Output ---------------------
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{seq_name}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(video_path, fourcc, seqinfo["frame_rate"], (seqinfo["width"], seqinfo["height"]))

    rng = np.random.default_rng(42)
    colour_cache = {}

    def get_colour(oid):
        if oid not in colour_cache:
            colour_cache[oid] = tuple(rng.integers(80, 255, 3).tolist())
        return colour_cache[oid]

    for frame_idx in range(seqinfo["seq_length"]):
        frame_id = frame_idx + 1
        img_path = os.path.join(image_dir, f"{frame_id:06d}{seqinfo['ext']}")
        img = cv2.imread(img_path)
        if img is None: 
            vw.write(np.zeros((seqinfo["height"], seqinfo["width"], 3), np.uint8))
            continue

        for row in results_by_frame.get(frame_id, []):
            _, tid, x, y, w, h, *_ = row
            c = get_colour(int(tid))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
            cv2.putText(img, f"ID {int(tid)}", (int(x), max(20, int(y) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        vw.write(img)

    vw.release()
    print(f"  → Video saved: {video_path}")

    # --------------------- Evaluate ---------------------
    metrics = evaluate_sequence(gt_by_frame, result_rows, args.eval_iou_thresh)
    metrics.update({"num_output_rows": len(result_rows), "result_path": result_path, "video_path": video_path})

    predictor.reset_state(inference_state)
    torch.cuda.empty_cache()

    return metrics


# ---------------------------------------------------------------------------
# Evaluation functions (kept from your original)
# ---------------------------------------------------------------------------

def xywh_to_tlbr(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def compute_iou_matrix(gt_bboxes, pred_bboxes):
    iou = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float64)
    for gi, gb in enumerate(gt_bboxes):
        for pi, pb in enumerate(pred_bboxes):
            iou[gi, pi] = bbox_iou(gb, pb)
    return iou


def evaluate_sequence(gt_by_frame, result_rows, iou_thresh=0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    pred_by_frame = collections.defaultdict(list)
    for row in result_rows:
        fid, tid = int(row[0]), int(row[1])
        pred_by_frame[fid].append({"id": tid, "bbox": np.array(row[2:6])})

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    for frame_id in all_frames:
        gt_objs = gt_by_frame.get(frame_id, [])
        pred_objs = pred_by_frame.get(frame_id, [])
        gt_ids = [g["id"] for g in gt_objs]
        pred_ids = [p["id"] for p in pred_objs]

        if gt_objs and pred_objs:
            iou_mat = compute_iou_matrix([g["bbox"] for g in gt_objs], [p["bbox"] for p in pred_objs])
            dist_mat = 1.0 - iou_mat
            dist_mat[iou_mat < iou_thresh] = np.nan
        else:
            dist_mat = np.full((len(gt_ids), len(pred_ids)), np.nan)

        acc.update(gt_ids, pred_ids, dist_mat)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_frames", "mota", "motp", "idf1", "num_switches", "num_fragmentations",
                                       "mostly_tracked", "mostly_lost", "num_false_positives", "num_misses"], name="seq")

    result = {col: float(summary[col].iloc[0]) if not isinstance(summary[col].iloc[0], str) else summary[col].iloc[0]
              for col in summary.columns}
    return result


def evaluate_all_sequences(mot_root, results_dir, seq_names, iou_thresh=0.5):
    # (Your original evaluate_all_sequences function - unchanged)
    # ... paste your original evaluate_all_sequences here if needed ...
    # For brevity, I'm assuming you keep it as-is from your previous code.
    pass   # Replace with your full evaluate_all_sequences if you need overall summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    results_dir = os.path.join(output_root, "mot_results")
    videos_dir = os.path.join(output_root, "videos")

    os.makedirs(output_root, exist_ok=True)

    if args.seq:
        seq_names = args.seq
    else:
        seq_names = sorted(e for e in os.listdir(args.mot_root) if os.path.isdir(os.path.join(args.mot_root, e)))

    print(f"Sequences to process: {seq_names}\n")

    per_sequence_metrics = {}
    for seq_name in seq_names:
        seq_path = os.path.join(args.mot_root, seq_name)
        if not os.path.isdir(seq_path): continue

        print(f"\n{'='*70}")
        print(f"Processing: {seq_name}")
        print(f"{'='*70}")

        per_sequence_metrics[seq_name] = run_tracker_on_sequence(
            seq_name, seq_path, args, results_dir, videos_dir
        )

    # Add your evaluate_all_sequences call here if needed
    print("\n✅ Done! Check outputs in:", output_root)


if __name__ == "__main__":
    main()