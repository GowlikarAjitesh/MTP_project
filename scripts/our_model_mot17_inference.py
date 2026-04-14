import argparse
import collections
import configparser
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

# Compatibility shim for newer NumPy with motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

mm.lap.default_solver = "scipy"
PEDESTRIAN_CLASS_ID = 1


@dataclass
class Detection:
    bbox: np.ndarray  # [x, y, w, h]
    score: float


@dataclass
class Track:
    obj_id: int
    last_bbox: np.ndarray  # [x, y, w, h]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM2 tracking on one MOT17 sequence using MOT17 public detections."
    )
    parser.add_argument(
        "--mot-root",
        default="~/datasets/MOT17/train",
        help="Root of MOT17 train sequences.",
    )
    parser.add_argument(
        "--seq",
        default="MOT17-02",
        help="MOT17 sequence base name, e.g. MOT17-02.",
    )
    parser.add_argument(
        "--det-set",
        choices=["FRCNN", "SDP", "DPM"],
        default="FRCNN",
        help="MOT17 detection set to use.",
    )
    parser.add_argument(
        "--output-root",
        default="~/phase2/mysam2_results",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2 config path relative to the sam2 package.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="~/phase2/mysam2/checkpoints/sam2.1_hiera_base_plus.pt",
        help="Path to SAM2 checkpoint.",
    )
    parser.add_argument(
        "--det-score-thresh",
        type=float,
        default=0.5,
        help="Minimum detection score to keep.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum age for unmatched tracks before removal.",
    )
    parser.add_argument(
        "--eval-iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold used for MOT metrics evaluation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process for testing.",
    )
    return parser.parse_args()


def load_seqinfo(seq_path: Path):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(seq_path / "seqinfo.ini")
    s = seqinfo["Sequence"]
    return {
        "name": s["name"],
        "im_dir": s["imDir"],
        "frame_rate": int(s["frameRate"]),
        "seq_length": int(s["seqLength"]),
        "width": int(s["imWidth"]),
        "height": int(s["imHeight"]),
        "ext": s["imExt"],
    }


def load_detections(det_path: Path, score_thresh: float):
    detections_by_frame = {}
    with open(det_path, "r") as f:
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


def load_ground_truth(gt_path: Path):
    gt_by_frame = {}
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            gt_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            conf = int(float(parts[6]))
            cls = int(float(parts[7]))
            if conf != 1 or cls != PEDESTRIAN_CLASS_ID:
                continue
            gt_by_frame.setdefault(frame_id, []).append(
                {"id": gt_id, "bbox": np.array([x, y, w, h], dtype=np.float32)}
            )
    return gt_by_frame


def mask_to_bbox(mask: np.ndarray):
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1], dtype=np.float32)


def bbox_iou(b1: np.ndarray, b2: np.ndarray):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2]); y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0.0 else 0.0


def masks_from_video_res_output(obj_ids, video_res_masks):
    frame_masks = {}
    if video_res_masks is None:
        return frame_masks
    masks_np = video_res_masks.detach().cpu().numpy()
    for i, oid in enumerate(obj_ids):
        mask_tensor = masks_np[i]
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor[0]
        frame_masks[int(oid)] = (mask_tensor > 0).astype(np.uint8)
    return frame_masks


def evaluate_sequence(gt_by_frame, result_rows, iou_thresh=0.5, frame_limit=None):
    acc = mm.MOTAccumulator(auto_id=True)
    pred_by_frame = collections.defaultdict(list)
    for row in result_rows:
        fid, tid, x, y, w, h = int(row[0]), int(row[1]), *row[2:6]
        pred_by_frame[fid].append({"id": tid, "bbox": np.array([x, y, w, h])})

    gt_frames = set(gt_by_frame.keys())
    pred_frames = set(pred_by_frame.keys())
    if frame_limit is not None:
        gt_frames = {fid for fid in gt_frames if fid <= frame_limit}
        pred_frames = {fid for fid in pred_frames if fid <= frame_limit}
    all_frames = sorted(gt_frames | pred_frames)
    for frame_id in all_frames:
        gt_objs = gt_by_frame.get(frame_id, [])
        pred_objs = pred_by_frame.get(frame_id, [])
        gt_ids = [g["id"] for g in gt_objs]
        pred_ids = [p["id"] for p in pred_objs]

        if gt_objs and pred_objs:
            iou_mat = np.zeros((len(gt_objs), len(pred_objs)), dtype=np.float64)
            for gi, gb in enumerate([g["bbox"] for g in gt_objs]):
                for pi, pb in enumerate([p["bbox"] for p in pred_objs]):
                    iou_mat[gi, pi] = bbox_iou(gb, pb)
            dist_mat = 1.0 - iou_mat
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
        metrics=["num_frames", "mota", "motp", "idf1", "num_switches",
                 "num_fragmentations", "mostly_tracked", "mostly_lost",
                 "num_false_positives", "num_misses"],
        name="seq",
    )
    result = {col: float(summary[col].iloc[0]) for col in summary.columns}
    return result


def run_sequence(args):
    mot_root = Path(args.mot_root).expanduser()
    if not mot_root.exists():
        raise RuntimeError(f"MOT root does not exist: {mot_root}")

    if args.seq.endswith("-FRCNN") or args.seq.endswith("-SDP") or args.seq.endswith("-DPM"):
        seq_name = args.seq
    else:
        seq_name = f"{args.seq}-{args.det_set}"
    seq_path = mot_root / seq_name
    if not seq_path.exists():
        raise RuntimeError(f"Sequence path not found: {seq_path}")

    seqinfo = load_seqinfo(seq_path)
    det_path = seq_path / "det" / "det.txt"
    gt_path = seq_path / "gt" / "gt.txt"
    image_dir = seq_path / seqinfo["im_dir"]
    max_frames = args.max_frames if args.max_frames is not None else seqinfo["seq_length"]
    seq_length = min(seqinfo["seq_length"], max_frames)

    if not det_path.exists():
        raise RuntimeError(f"Detection file not found: {det_path}")
    if not gt_path.exists():
        raise RuntimeError(f"GT file not found: {gt_path}")
    if not image_dir.exists():
        raise RuntimeError(f"Image directory not found: {image_dir}")

    detections = load_detections(det_path, args.det_score_thresh)
    gt_by_frame = load_ground_truth(gt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    predictor = build_sam2_video_predictor(
        args.sam2_config,
        str(Path(args.sam2_checkpoint).expanduser()),
        device=device,
    )
    inference_state = predictor.init_state(video_path=str(image_dir))
    if hasattr(predictor, "trajectory_manager"):
        predictor.trajectory_manager.conf_thresh = args.det_score_thresh
        predictor.trajectory_manager.lost_tolerance = args.max_age

    next_obj_id = 1
    active_tracks = []
    result_rows = []
    results_by_frame = collections.defaultdict(list)

    output_root = Path(args.output_root).expanduser() / seq_name
    results_dir = output_root / "results"
    videos_dir = output_root / "videos"
    masks_dir = output_root / "masks"
    overlays_dir = output_root / "mask_overlays"
    for d in [results_dir, videos_dir, masks_dir, overlays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Tracking {seq_name} ({seq_length} frames, max_frames={args.max_frames})")

    rng = np.random.default_rng(42)
    colour_cache = {}
    def get_colour(oid):
        if oid not in colour_cache:
            colour_cache[oid] = tuple(rng.integers(80, 255, 3).tolist())
        return colour_cache[oid]

    for frame_idx in range(seq_length):
        frame_id = frame_idx + 1
        frame_dets = detections.get(frame_id, [])
        if frame_dets:
            det_array = np.stack([
                [*d.bbox[:2], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3], d.score]
                for d in frame_dets
            ], axis=0).astype(np.float32)
        else:
            det_array = np.empty((0, 5), dtype=np.float32)

        # Bootstrap tracking when no objects are active yet.
        if not active_tracks and len(det_array) > 0:
            for det in frame_dets:
                x, y, w, h = det.bbox
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=next_obj_id,
                    box=box,
                )
                active_tracks.append(Track(obj_id=next_obj_id, last_bbox=det.bbox.copy()))
                next_obj_id += 1

        frame_masks = {}
        if active_tracks:
            for out_fidx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                detections=det_array,
            ):
                if out_fidx != frame_idx:
                    continue
                frame_masks = masks_from_video_res_output(out_obj_ids, out_mask_logits)

            sam2mot_update = inference_state.get("sam2mot_last_update") or {}
            if sam2mot_update.get("frame_idx") == frame_idx:
                removed_ids = set(sam2mot_update.get("to_remove", []))
                if removed_ids:
                    for oid in list(removed_ids):
                        predictor.remove_object(
                            inference_state=inference_state,
                            obj_id=oid,
                            strict=False,
                            need_output=False,
                        )
                    active_tracks = [t for t in active_tracks if t.obj_id not in removed_ids]
                    for oid in removed_ids:
                        frame_masks.pop(oid, None)

                for new_obj in sam2mot_update.get("new_objects", []):
                    box = np.array(new_obj["box"], dtype=np.float32)
                    _, obj_ids_after_add, video_res_masks_after_add = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=next_obj_id,
                        box=box,
                    )
                    frame_masks = masks_from_video_res_output(obj_ids_after_add, video_res_masks_after_add)
                    x1, y1, x2, y2 = box
                    active_tracks.append(
                        Track(
                            obj_id=next_obj_id,
                            last_bbox=np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
                        )
                    )
                    next_obj_id += 1

        # Keep only tracks that still exist in the predictor state.
        current_obj_ids = set(inference_state["obj_ids"])
        active_tracks = [t for t in active_tracks if t.obj_id in current_obj_ids]

        # Save masks for this frame
        if frame_masks:
            ids = np.array(list(frame_masks.keys()), dtype=np.int32)
            masks = np.stack([frame_masks[oid] for oid in ids], axis=0)
            np.savez_compressed(masks_dir / f"{frame_id:06d}.npz", ids=ids, masks=masks)
            image_path = image_dir / f"{frame_id:06d}{seqinfo['ext']}"
            image = cv2.imread(str(image_path))
            if image is None:
                image = np.zeros((seqinfo['height'], seqinfo['width'], 3), dtype=np.uint8)
            overlay = image.copy()
            for oid, mask in frame_masks.items():
                color = np.array(get_colour(oid), dtype=np.uint8)
                mask_rgb = np.stack([mask] * 3, axis=-1)
                blended = (overlay.astype(np.float32) * 0.5 + color.astype(np.float32) * 0.5).astype(np.uint8)
                overlay = np.where(mask_rgb, blended, overlay)
            cv2.imwrite(str(overlays_dir / f"{frame_id:06d}.png"), overlay)

        for trk in active_tracks:
            mask = frame_masks.get(trk.obj_id)
            if mask is not None:
                bbox = mask_to_bbox(mask)
                if bbox is not None:
                    trk.last_bbox = bbox

            x, y, w, h = trk.last_bbox
            row = [frame_id, trk.obj_id, float(x), float(y), float(w), float(h), 1.0, -1, -1, -1]
            result_rows.append(row)
            results_by_frame[frame_id].append(row)

        if frame_id % 50 == 0 or frame_id == seqinfo["seq_length"]:
            print(f"  Frame {frame_id}/{seqinfo['seq_length']} | active tracks: {len(active_tracks)}")

    result_path = results_dir / f"{seq_name}.txt"
    with open(result_path, "w") as f:
        for row in result_rows:
            f.write(",".join(map(str, row)) + "\n")

    video_path = videos_dir / f"{seq_name}.avi"
    vw = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"XVID"),
        seqinfo["frame_rate"], (seqinfo["width"], seqinfo["height"])
    )
    for frame_idx in range(seq_length):
        frame_id = frame_idx + 1
        image_path = image_dir / f"{frame_id:06d}{seqinfo['ext']}"
        frame = cv2.imread(str(image_path))
        if frame is None:
            frame = np.zeros((seqinfo["height"], seqinfo["width"], 3), dtype=np.uint8)
        for row in results_by_frame.get(frame_id, []):
            _, tid, x, y, w, h, *_ = row
            c = get_colour(int(tid))
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
            cv2.putText(frame, f"ID{int(tid)}", (int(x), max(18, int(y) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        vw.write(frame)
    vw.release()

    metrics = evaluate_sequence(gt_by_frame, result_rows, args.eval_iou_thresh, frame_limit=seq_length)
    metrics.update({
        "seq_name": seq_name,
        "result_path": str(result_path),
        "video_path": str(video_path),
        "masks_dir": str(masks_dir),
        "overlay_dir": str(overlays_dir),
        "num_rows": len(result_rows),
    })

    metrics_path = output_root / f"{seq_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Finished {seq_name}")
    print(f"  MOT result file: {result_path}")
    print(f"  Video output:    {video_path}")
    print(f"  Mask arrays:     {masks_dir}")
    print(f"  Mask overlays:   {overlays_dir}")
    print(f"  Metrics JSON:    {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    args = parse_args()
    run_sequence(args)


if __name__ == "__main__":
    main()
