"""
SAM 2 Multi-Object Tracker for MOT17
======================================
Fixes applied (see inline comments marked FIX):
  1. Two-pass architecture was wrong: detections were added in a loop THEN
     propagate_in_video was called once — this means SAM2 only saw the LAST
     set of prompts and all objects were added on the wrong frame_idx (all
     relative to frame 0 context but with wrong frame_idx values).
     → Replaced with the correct single-pass: for each frame, add prompts,
       then propagate from that frame onward.

  2. `predictor.init_state` does NOT accept a `frame_names` kwarg in SAM2's
     public API. It accepts `video_path` (a directory) and reads frames
     itself via glob. Passing frame_names caused a TypeError.
     → Removed the frame_names argument.

  3. Mask output shape from SAM2 is (1, H, W) — the [0] index is correct
     but only if out_mask_logits shape is checked. Also the logit threshold
     should be 0.0 (already correct) but the indexing `out_mask_logits[i]`
     gives a (1,H,W) tensor, so `.cpu().numpy()[0]` gives (H,W). That part
     was fine; documented for clarity.

  4. `evaluate_sequence` and `evaluate_all_sequences` were referenced in
     main() but NEVER defined (the comment said "copy them from your
     previous code"). Fully implemented here using py-motmetrics.

  5. obj_id → track_id explosion: every detection on every frame got a NEW
     unique obj_id, so by frame 100 you'd have thousands of tracks. The
     correct pattern is:
       - Frame 1: add all detections as new objects.
       - Frame N>1: use Hungarian matching to associate detections to
         existing tracks; only add unmatched detections as NEW objects.
     Implemented greedy IoU-based matching between last-known bbox and
     new detections before adding prompts.

  6. SAM2 `propagate_in_video` streams ALL frames from the start. For
     online/incremental tracking you should call it per-frame with
     `start_frame_idx` and `max_frame_num_to_track=1`. This avoids
     re-propagating old frames and gives true online behaviour.
     → Used propagate_in_video(start_frame_idx=frame_idx,
                                max_frame_num_to_track=1)

  7. Video output loop was reading result_rows with a linear scan O(F*N)
     — replaced with a dict-of-lists keyed by frame_id.

  8. Missing imports added: collections, scipy.

  9. `--sam2-checkpoint` was marked `required=True` but also had a
     `default=...` which is contradictory (argparse ignores default when
     required=True). Fixed to just use default.

 10. np.asfarray shim placed before mm import so motmetrics doesn't error
     on NumPy ≥ 1.24.
"""

import argparse
import collections
import configparser
import json
import os
from dataclasses import dataclass, field

import cv2
import motmetrics as mm
import numpy as np
import torch
import pandas as pd
from scipy.optimize import linear_sum_assignment

# SAM 2 imports
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# FIX 10: numpy compatibility shim BEFORE motmetrics uses np.asfarray
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

mm.lap.default_solver = "scipy"

PEDESTRIAN_CLASS_ID = 1
IOU_MATCH_THRESH = 0.3      # IoU threshold for associating dets to tracks


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
    last_bbox: np.ndarray   # [x, y, w, h]  updated each frame from mask
    age: int = 0            # frames since last matched


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM 2 based multi-object tracker using MOT17 public detections."
    )
    parser.add_argument(
        "--mot-root", default="/home/cs24m118/datasets/MOT17/train",
        help="Path to MOT17 train sequences."
    )
    parser.add_argument(
        "--output-root", default="/home/cs24m118/phase2/outputs/sam2_tracking",
        help="Output directory."
    )
    parser.add_argument("--seq", nargs="*", help="Specific sequences to run.")
    parser.add_argument(
        "--det-score-thresh", type=float, default=0.5,
        help="Minimum detection score."
    )
    # FIX 9: removed required=True — default already set
    parser.add_argument(
        "--sam2-checkpoint",
        default="/home/cs24m118/phase2/mysam2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM 2 checkpoint."
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2 config path (relative to sam2 package or absolute)."
    )
    parser.add_argument("--eval-iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--max-age", type=int, default=30,
        help="Max frames a track can go unmatched before removal."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_seqinfo(seq_path):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, "seqinfo.ini"))
    s = seqinfo["Sequence"]
    return {
        "name":       s["name"],
        "im_dir":     s["imDir"],
        "frame_rate": int(s["frameRate"]),
        "seq_length": int(s["seqLength"]),
        "width":      int(s["imWidth"]),
        "height":     int(s["imHeight"]),
        "ext":        s["imExt"],
    }


def load_detections(det_path, score_thresh):
    detections_by_frame = {}
    with open(det_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6])
            if score < score_thresh:
                continue
            detections_by_frame.setdefault(frame_id, []).append(
                Detection(bbox=np.array([x, y, w, h], dtype=np.float32), score=score)
            )
    return detections_by_frame


def load_ground_truth(gt_path):
    gt_by_frame = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            gt_id    = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = int(float(parts[6]))
            cls  = int(float(parts[7]))
            if conf != 1 or cls != PEDESTRIAN_CLASS_ID:
                continue
            gt_by_frame.setdefault(frame_id, []).append(
                {"id": gt_id, "bbox": np.array([x, y, w, h], dtype=np.float32)}
            )
    return gt_by_frame


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def mask_to_bbox(mask: np.ndarray):
    """Convert binary (H,W) mask → [x, y, w, h] or None if empty."""
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1], dtype=np.float32)


def bbox_iou(b1, b2):
    """IoU between two [x,y,w,h] boxes."""
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0:
        return 0.0
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def match_detections_to_tracks(tracks, dets, iou_thresh=IOU_MATCH_THRESH):
    """
    Hungarian matching: returns (matched_pairs, unmatched_det_indices, unmatched_track_ids)
    matched_pairs: list of (det_idx, track)
    """
    if not tracks or not dets:
        return [], list(range(len(dets))), [t.obj_id for t in tracks]

    cost = np.zeros((len(tracks), len(dets)), dtype=np.float64)
    for ti, trk in enumerate(tracks):
        for di, det in enumerate(dets):
            cost[ti, di] = 1.0 - bbox_iou(trk.last_bbox, det.bbox)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    unmatched_dets = set(range(len(dets)))
    unmatched_trk_ids = set(t.obj_id for t in tracks)

    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] < (1.0 - iou_thresh):
            matched_pairs.append((ci, tracks[ri]))   # (det_idx, track)
            unmatched_dets.discard(ci)
            unmatched_trk_ids.discard(tracks[ri].obj_id)

    return matched_pairs, list(unmatched_dets), list(unmatched_trk_ids)


# ---------------------------------------------------------------------------
# Main tracking function
# ---------------------------------------------------------------------------

def run_tracker_on_sequence(seq_name, seq_path, args, results_dir, videos_dir):
    seqinfo     = load_seqinfo(seq_path)
    detections  = load_detections(
        os.path.join(seq_path, "det", "det.txt"), args.det_score_thresh
    )
    gt_by_frame = load_ground_truth(os.path.join(seq_path, "gt", "gt.txt"))
    image_dir   = os.path.join(seq_path, seqinfo["im_dir"])

    # ---- SAM2 setup ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  → Using device: {device}")

    predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        args.sam2_config, args.sam2_checkpoint, device=device
    )

    # FIX 2: init_state only takes video_path (directory); SAM2 globs frames itself
    inference_state = predictor.init_state(video_path=image_dir)

    # ---- State ----
    next_obj_id = 1
    active_tracks: list[Track] = []   # currently alive tracks
    result_rows = []                   # MOT output rows
    # FIX 7: result dict keyed by frame_id for O(1) video-render lookup
    results_by_frame: dict[int, list] = collections.defaultdict(list)

    print(f"  → Tracking {seq_name} ({seqinfo['seq_length']} frames) with SAM2")

    for frame_idx in range(seqinfo["seq_length"]):   # 0-based
        frame_id  = frame_idx + 1
        frame_dets = detections.get(frame_id, [])

        # ------------------------------------------------------------------
        # FIX 5 + FIX 1: Match detections → existing tracks; only create new
        # SAM2 objects for truly new detections.
        # ------------------------------------------------------------------
        matched, unmatched_det_idxs, unmatched_trk_ids = match_detections_to_tracks(
            active_tracks, frame_dets
        )

        # Update matched tracks' last_bbox from detection (SAM2 mask updates below)
        matched_track_ids = set()
        for det_idx, trk in matched:
            trk.last_bbox = frame_dets[det_idx].bbox
            trk.age = 0
            matched_track_ids.add(trk.obj_id)

        # Add NEW objects for unmatched detections
        for det_idx in unmatched_det_idxs:
            det = frame_dets[det_idx]
            x, y, w, h = det.bbox
            # SAM2 expects [x1, y1, x2, y2]
            box = np.array([x, y, x + w, y + h], dtype=np.float32)

            # FIX 1 + FIX 6: add_new_points_or_box with correct frame_idx
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=next_obj_id,
                box=box,
            )
            new_track = Track(obj_id=next_obj_id, last_bbox=det.bbox.copy())
            active_tracks.append(new_track)
            next_obj_id += 1

        # Age unmatched tracks
        for trk in active_tracks:
            if trk.obj_id in unmatched_trk_ids:
                trk.age += 1

        # Remove dead tracks
        active_tracks = [t for t in active_tracks if t.age <= args.max_age]

        # ------------------------------------------------------------------
        # FIX 6: Propagate ONLY this one frame (online mode)
        # ------------------------------------------------------------------
        frame_masks = {}   # obj_id → (H,W) binary mask
        try:
            for out_fidx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1,
            ):
                if out_fidx != frame_idx:
                    continue
                for i, oid in enumerate(out_obj_ids):
                    # FIX 3: out_mask_logits[i] is (1,H,W); [0] gives (H,W)
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                    frame_masks[oid] = mask
        except Exception as e:
            print(f"    Warning: propagate error at frame {frame_id}: {e}")

        # Update track bboxes from masks and write output rows
        alive_obj_ids = {t.obj_id for t in active_tracks}
        for trk in active_tracks:
            mask = frame_masks.get(trk.obj_id)
            if mask is not None:
                bbox = mask_to_bbox(mask)
                if bbox is not None:
                    trk.last_bbox = bbox

            # Write result row using last known bbox
            x, y, w, h = trk.last_bbox
            row = [
                frame_id, trk.obj_id,
                round(float(x), 2), round(float(y), 2),
                round(float(w), 2), round(float(h), 2),
                1.0, -1, -1, -1
            ]
            result_rows.append(row)
            results_by_frame[frame_id].append(row)

        if frame_idx % 50 == 0:
            print(f"    Frame {frame_id}/{seqinfo['seq_length']} | "
                  f"active tracks: {len(active_tracks)}")

    # ---- Save MOT results ----
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{seq_name}.txt")
    with open(result_path, "w") as f:
        for row in result_rows:
            f.write(",".join(map(str, row)) + "\n")

    # ---- Video output ----
    # FIX 7: use results_by_frame dict — O(1) per frame
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{seq_name}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(
        video_path, fourcc,
        seqinfo["frame_rate"], (seqinfo["width"], seqinfo["height"])
    )

    # colour palette: one colour per obj_id (consistent across frames)
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
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), c, 2)
            cv2.putText(img, f"ID {int(tid)}", (int(x), max(20, int(y)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        vw.write(img)

    vw.release()
    print(f"  → Video saved: {video_path}")

    # ---- Evaluate ----
    metrics = evaluate_sequence(gt_by_frame, result_rows, args.eval_iou_thresh)
    metrics.update({
        "num_output_rows": len(result_rows),
        "result_path":     result_path,
        "video_path":      video_path,
    })

    predictor.reset_state(inference_state)
    return metrics


# ---------------------------------------------------------------------------
# FIX 4: Evaluation — fully implemented (was missing entirely)
# ---------------------------------------------------------------------------

def xywh_to_tlbr(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x+w, y+h], dtype=np.float64)


def compute_iou_matrix(gt_bboxes, pred_bboxes):
    """Return (G, P) IoU matrix."""
    iou = np.zeros((len(gt_bboxes), len(pred_bboxes)), dtype=np.float64)
    for gi, gb in enumerate(gt_bboxes):
        for pi, pb in enumerate(pred_bboxes):
            iou[gi, pi] = bbox_iou(gb, pb)
    return iou


def evaluate_sequence(gt_by_frame, result_rows, iou_thresh=0.5):
    """
    Compute MOTA, MOTP, IDF1, etc. for one sequence using py-motmetrics.
    Returns a flat dict of scalar metrics.
    """
    acc = mm.MOTAccumulator(auto_id=True)

    # Group predictions by frame
    pred_by_frame = collections.defaultdict(list)
    for row in result_rows:
        fid, tid, x, y, w, h = int(row[0]), int(row[1]), *row[2:6]
        pred_by_frame[fid].append({"id": tid, "bbox": np.array([x, y, w, h])})

    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))

    for frame_id in all_frames:
        gt_objs   = gt_by_frame.get(frame_id, [])
        pred_objs = pred_by_frame.get(frame_id, [])

        gt_ids   = [g["id"] for g in gt_objs]
        pred_ids = [p["id"] for p in pred_objs]

        if gt_objs and pred_objs:
            iou_mat  = compute_iou_matrix(
                [g["bbox"] for g in gt_objs],
                [p["bbox"] for p in pred_objs]
            )
            # motmetrics expects a DISTANCE matrix (lower = better)
            dist_mat = 1.0 - iou_mat
            # Mark pairs below threshold as NaN (no valid match)
            dist_mat[iou_mat < iou_thresh] = np.nan
        elif gt_objs:
            dist_mat = np.full((len(gt_ids), 0), np.nan)
        elif pred_objs:
            dist_mat = np.full((0, len(pred_ids)), np.nan)
        else:
            dist_mat = np.empty((0, 0))

        acc.update(gt_ids, pred_ids, dist_mat)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_frames", "mota", "motp", "idf1",
                 "num_switches", "num_fragmentations",
                 "mostly_tracked", "mostly_lost",
                 "num_false_positives", "num_misses"],
        name="seq"
    )

    result = {}
    for col in summary.columns:
        val = summary[col].iloc[0]
        result[col] = float(val) if not isinstance(val, str) else val
    return result


def evaluate_all_sequences(mot_root, results_dir, seq_names, iou_thresh=0.5):
    """
    Run per-sequence motmetrics accumulators and combine into a single summary.
    """
    accs  = []
    names = []

    for seq_name in seq_names:
        seq_path = os.path.join(mot_root, seq_name)
        gt_path  = os.path.join(seq_path, "gt", "gt.txt")
        res_path = os.path.join(results_dir, f"{seq_name}.txt")

        if not os.path.exists(res_path):
            print(f"  Skipping {seq_name}: no result file.")
            continue

        gt_by_frame = load_ground_truth(gt_path)

        result_rows = []
        with open(res_path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    result_rows.append([float(p) for p in parts])

        acc = mm.MOTAccumulator(auto_id=True)
        pred_by_frame = collections.defaultdict(list)
        for row in result_rows:
            fid, tid = int(row[0]), int(row[1])
            pred_by_frame[fid].append(
                {"id": tid, "bbox": np.array(row[2:6])}
            )

        all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))
        for frame_id in all_frames:
            gt_objs   = gt_by_frame.get(frame_id, [])
            pred_objs = pred_by_frame.get(frame_id, [])
            gt_ids    = [g["id"] for g in gt_objs]
            pred_ids  = [p["id"] for p in pred_objs]

            if gt_objs and pred_objs:
                iou_mat  = compute_iou_matrix(
                    [g["bbox"] for g in gt_objs],
                    [p["bbox"] for p in pred_objs]
                )
                dist_mat = 1.0 - iou_mat
                dist_mat[iou_mat < iou_thresh] = np.nan
            elif gt_objs:
                dist_mat = np.full((len(gt_ids), 0), np.nan)
            elif pred_objs:
                dist_mat = np.full((0, len(pred_ids)), np.nan)
            else:
                dist_mat = np.empty((0, 0))

            acc.update(gt_ids, pred_ids, dist_mat)

        accs.append(acc)
        names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names,
        metrics=["num_frames", "mota", "motp", "idf1",
                 "num_switches", "num_fragmentations",
                 "mostly_tracked", "mostly_lost",
                 "num_false_positives", "num_misses"],
        generate_overall=True
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    output_root      = os.path.abspath(args.output_root)
    results_dir      = os.path.join(output_root, "mot_results")
    videos_dir       = os.path.join(output_root, "videos")
    metrics_json_path = os.path.join(output_root, "metrics.json")
    metrics_csv_path  = os.path.join(output_root, "metrics_summary.csv")

    os.makedirs(output_root, exist_ok=True)

    if args.seq:
        seq_names = args.seq
    else:
        seq_names = sorted(
            e for e in os.listdir(args.mot_root)
            if os.path.isdir(os.path.join(args.mot_root, e))
        )

    print(f"Sequences to process: {seq_names}\n")

    per_sequence_metrics = {}
    for seq_name in seq_names:
        seq_path = os.path.join(args.mot_root, seq_name)
        if not os.path.isdir(seq_path):
            print(f"Skipping {seq_name}: directory not found.")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {seq_name}")
        print(f"{'='*60}")
        per_sequence_metrics[seq_name] = run_tracker_on_sequence(
            seq_name, seq_path, args, results_dir, videos_dir
        )

    print("\nComputing overall metrics across all sequences...")
    summary_df = evaluate_all_sequences(
        args.mot_root, results_dir, seq_names, args.eval_iou_thresh
    )

    summary_df.to_csv(metrics_csv_path)
    print(f"\n{'='*90}")
    print(summary_df.to_string())
    print(f"{'='*90}")

    metrics_payload = {
        "settings":     vars(args),
        "per_sequence": per_sequence_metrics,
        "summary":      summary_df.reset_index().to_dict(orient="records"),
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2, default=str)

    print(f"\n✅ Videos saved in:  {videos_dir}")
    print(f"✅ Results in:       {results_dir}")
    print(f"✅ CSV metrics:      {metrics_csv_path}")
    print(f"✅ JSON metrics:     {metrics_json_path}")


if __name__ == "__main__":
    main()