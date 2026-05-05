import argparse
import gc
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore", message="A NumPy version >=", category=UserWarning)

# motmetrics still calls np.asfarray, which was removed in NumPy 2.0.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda array_like, dtype=float: np.asarray(array_like, dtype=dtype)

import motmetrics as mm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = PROJECT_ROOT / "sam2"
if str(SAM2_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_ROOT))

from sam2.build_sam import build_sam2_video_predictor


DEFAULT_SEQUENCE = "MOT17-02-FRCNN"
DEFAULT_DATASET_ROOT = Path("/home/cs24m118/datasets/MOT17/train")
DEFAULT_CONFIG = PROJECT_ROOT / "sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "sam2/checkpoints/sam2.1_hiera_small.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track all frame-1 objects in one MOT17 sequence with SAM2/SAMURAI/DAM, save masks, video, and metrics."
    )
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs" / "single_sequence_tracking")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means use all frames")
    parser.add_argument("--max_objects", type=int, default=0, help="0 means use all frame-1 detections")
    parser.add_argument(
        "--redetect_interval",
        type=int,
        default=30,
        help="Run detection-based track birth every N frames by restarting SAM2 from that frame",
    )
    parser.add_argument(
        "--birth_iou_threshold",
        type=float,
        default=0.5,
        help="Do not spawn a new track if a detection overlaps an active track above this IoU",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="0 means read from seqinfo.ini or fall back to 30")
    parser.add_argument("--save_per_object_masks", action="store_true", default=True)
    parser.add_argument("--dam_mode", action="store_true", default=True)
    parser.add_argument("--samurai_mode", action="store_true", default=True)
    parser.add_argument("--memory_update_stride", type=int, default=5)
    parser.add_argument("--anchor_threshold", type=float, default=0.7)
    parser.add_argument("--iou_threshold", type=float, default=0.8)
    parser.add_argument("--area_threshold", type=float, default=0.2)
    parser.add_argument("--median_window", type=int, default=10)
    parser.add_argument("--stable_frames_threshold", type=int, default=15)
    parser.add_argument("--stable_ious_threshold", type=float, default=0.3)
    parser.add_argument("--memory_bank_iou_threshold", type=float, default=0.7)
    parser.add_argument("--memory_bank_obj_score_threshold", type=float, default=0.5)
    parser.add_argument("--memory_bank_kf_score_threshold", type=float, default=0.3)
    parser.add_argument("--kf_score_weight", type=float, default=0.2)
    return parser.parse_args()


def read_mot_boxes(txt_path: Path, conf_thresh: float, is_gt: bool):
    boxes_by_frame = defaultdict(list)
    if not txt_path.exists():
        return boxes_by_frame

    with txt_path.open("r") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            class_id = int(float(parts[7])) if len(parts) > 7 else 1
            if is_gt:
                if conf <= 0 or class_id != 1:
                    continue
            elif conf < conf_thresh:
                continue

            boxes_by_frame[frame].append(
                {
                    "track_id": track_id,
                    "conf": conf,
                    "class_id": class_id,
                    "bbox_xyxy": [x, y, x + w, y + h],
                }
            )
    return boxes_by_frame


def clamp_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def bbox_from_binary_mask(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def color_for_id(obj_id):
    rng = np.random.default_rng(obj_id * 1337 + 17)
    c = rng.integers(32, 255, size=(3,), dtype=np.uint8)
    return int(c[0]), int(c[1]), int(c[2])


def read_fps(seq_dir: Path, fallback_fps: float):
    if fallback_fps > 0:
        return fallback_fps
    seqinfo_path = seq_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        for line in seqinfo_path.read_text().splitlines():
            if line.startswith("frameRate="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    break
    return 30.0


def is_cuda_oom(exc: Exception):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text or "cuda out of memory" in text


def cleanup_after_failure():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_hydra_config_name(config_path: Path | str):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_str = str(config_path).replace("\\", "/")
        return config_str

    config_roots = [
        PROJECT_ROOT / "sam2/sam2/configs",
        PROJECT_ROOT / "sam2/configs",
        PROJECT_ROOT / "configs",
    ]
    for root in config_roots:
        try:
            rel = config_path.relative_to(root)
            rel_str = str(rel).replace("\\", "/")
            if root.name == "configs":
                return f"configs/{rel_str}"
            return rel_str
        except ValueError:
            continue

    raise ValueError(f"Could not convert config path to Hydra config name: {config_path}")


def build_predictor(args, device):
    return build_sam2_video_predictor(
        config_file=to_hydra_config_name(args.config),
        ckpt_path=str(args.checkpoint),
        device=device,
        dam_mode=args.dam_mode,
        memory_update_stride=args.memory_update_stride,
        anchor_threshold=args.anchor_threshold,
        iou_threshold=args.iou_threshold,
        area_threshold=args.area_threshold,
        median_window=args.median_window,
        samurai_mode=args.samurai_mode,
        stable_frames_threshold=args.stable_frames_threshold,
        stable_ious_threshold=args.stable_ious_threshold,
        memory_bank_iou_threshold=args.memory_bank_iou_threshold,
        memory_bank_obj_score_threshold=args.memory_bank_obj_score_threshold,
        memory_bank_kf_score_threshold=args.memory_bank_kf_score_threshold,
        kf_score_weight=args.kf_score_weight,
    )


def resolve_repo_path(path_value: Path, candidates):
    candidate_path = Path(path_value)
    if candidate_path.exists():
        return candidate_path

    for candidate in candidates:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate

    return candidate_path


def trim_inference_state(inference_state, max_frames, start_frame=0):
    total_frames = inference_state["num_frames"]
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} is outside [0, {total_frames})")

    end_frame = total_frames if max_frames <= 0 else min(total_frames, start_frame + max_frames)
    if start_frame == 0 and end_frame == total_frames:
        return

    inference_state["images"] = inference_state["images"][start_frame:end_frame]
    inference_state["num_frames"] = end_frame - start_frame
    inference_state["cached_features"] = {}


def sort_detections_for_frame(frame_dets, width, height, limit=0):
    detections = [
        {
            **det,
            "bbox_xyxy": clamp_box_xyxy(det["bbox_xyxy"], width, height),
        }
        for det in frame_dets
    ]
    detections.sort(
        key=lambda det: (det["bbox_xyxy"][2] - det["bbox_xyxy"][0]) * (det["bbox_xyxy"][3] - det["bbox_xyxy"][1]),
        reverse=True,
    )
    if limit > 0:
        detections = detections[:limit]
    return detections


def greedy_match_boxes(track_boxes, detections, iou_threshold):
    candidates = []
    for track_id, track_box in track_boxes.items():
        for det_idx, det in enumerate(detections):
            iou = iou_xyxy(track_box, det["bbox_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, track_id, det_idx))

    candidates.sort(reverse=True)
    matches = []
    matched_tracks = set()
    matched_det_indices = set()
    for iou, track_id, det_idx in candidates:
        if track_id in matched_tracks or det_idx in matched_det_indices:
            continue
        matched_tracks.add(track_id)
        matched_det_indices.add(det_idx)
        matches.append((track_id, det_idx, iou))

    unmatched_tracks = [track_id for track_id in track_boxes if track_id not in matched_tracks]
    unmatched_det_indices = [det_idx for det_idx in range(len(detections)) if det_idx not in matched_det_indices]
    return matches, unmatched_tracks, unmatched_det_indices


def build_frame_outputs(obj_ids, video_res_masks):
    frame_pred_boxes = {}
    frame_pred_masks = {}
    active_objects = 0

    for i, obj_id in enumerate(obj_ids):
        binary_mask = (video_res_masks[i, 0] > 0).detach().cpu().numpy().astype(np.uint8)
        if binary_mask.sum() == 0:
            continue

        bbox = bbox_from_binary_mask(binary_mask)
        if bbox is None:
            continue

        active_objects += 1
        frame_pred_boxes[obj_id] = bbox
        frame_pred_masks[obj_id] = binary_mask

    return frame_pred_boxes, frame_pred_masks, active_objects


def write_frame_artifacts(
    frame_idx,
    frame_path,
    frame_pred_boxes,
    frame_pred_masks,
    active_objects,
    packed_mask_dir,
    per_object_mask_dir,
    writer,
    mot_lines,
    frame_level_stats,
    per_object_stats,
    gt_boxes_by_frame,
    mot_acc,
    save_per_object_masks,
):
    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise RuntimeError(f"Failed to read frame for video rendering: {frame_path}")

    height, width = bgr.shape[:2]
    packed_mask = np.zeros((height, width), dtype=np.uint16)
    overlay = bgr.copy()

    for obj_id in sorted(frame_pred_boxes):
        binary_mask = frame_pred_masks[obj_id]
        bbox = frame_pred_boxes[obj_id]
        per_object_stats.setdefault(obj_id, {"frames_present": 0, "pixel_sum": 0})
        per_object_stats[obj_id]["frames_present"] += 1
        per_object_stats[obj_id]["pixel_sum"] += int(binary_mask.sum())

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        mot_lines.append(f"{frame_idx + 1},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

        if save_per_object_masks:
            obj_dir = per_object_mask_dir / f"obj_{obj_id}"
            obj_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(obj_dir / f"{frame_idx:06d}.png"), binary_mask * 255)

        write_region = (packed_mask == 0) & binary_mask.astype(bool)
        packed_mask[write_region] = obj_id

        color = color_for_id(obj_id)
        color_layer = np.zeros_like(overlay, dtype=np.uint8)
        color_layer[binary_mask.astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0.0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"id:{obj_id}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(packed_mask_dir / f"{frame_idx:06d}.png"), packed_mask)
    writer.write(overlay)
    frame_level_stats.append({"frame_idx": frame_idx, "active_objects": active_objects})

    gt_frame_boxes = gt_boxes_by_frame.get(frame_idx + 1, [])
    if mot_acc is not None:
        gt_ids = [gt["track_id"] for gt in gt_frame_boxes]
        gt_xywh = [xyxy_to_xywh(gt["bbox_xyxy"]) for gt in gt_frame_boxes]
        pred_ids = sorted(frame_pred_boxes.keys())
        pred_xywh = [xyxy_to_xywh(frame_pred_boxes[obj_id]) for obj_id in pred_ids]
        distance_matrix = mm.distances.iou_matrix(gt_xywh, pred_xywh, max_iou=0.5)
        mot_acc.update(gt_ids, pred_ids, distance_matrix)


def prepare_segment_state(predictor, video_dir, start_frame, segment_frames):
    inference_state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )
    trim_inference_state(inference_state, segment_frames, start_frame=start_frame)
    return inference_state


def write_summary(summary_path: Path, metrics):
    lines = [
        f"Sequence: {metrics['sequence']}",
        f"Device used: {metrics['device_used']}",
        f"Frames processed: {metrics['frames_processed']}",
        f"Tracks created: {metrics['objects_prompted']}",
        f"Redetect interval: {metrics.get('redetect_interval', 'n/a')}",
        f"Objects with any mask: {metrics['objects_with_any_mask']}",
        f"Average active objects per frame: {metrics['avg_active_objects_per_frame']:.2f}",
        f"Video: {metrics['video_path']}",
        f"Packed masks: {metrics['packed_mask_dir']}",
        f"Per-object masks: {metrics['per_object_mask_dir']}",
        f"MOT results txt: {metrics['mot_txt_path']}",
    ]

    mot_metrics = metrics.get("mot_metrics", {})
    if mot_metrics.get("status") == "success":
        lines.extend(
            [
                "",
                "MOT metrics:",
                f"MOTA: {mot_metrics['mota_percent']:.2f}%",
                f"IDF1: {mot_metrics['idf1_percent']:.2f}%",
                f"Precision: {mot_metrics['precision_percent']:.2f}%",
                f"Recall: {mot_metrics['recall_percent']:.2f}%",
                f"ID Switches: {mot_metrics['id_switches']}",
                f"Mostly Tracked: {mot_metrics['mostly_tracked']} ({mot_metrics['mostly_tracked_ratio_percent']:.2f}%)",
                f"Mostly Lost: {mot_metrics['mostly_lost']} ({mot_metrics['mostly_lost_ratio_percent']:.2f}%)",
            ]
        )
    else:
        lines.extend(["", f"MOT metrics: {mot_metrics.get('status', 'unavailable')}"])

    summary_path.write_text("\n".join(lines) + "\n")


def run_sequence(args, device):
    seq_dir = args.dataset_root / args.sequence
    video_dir = seq_dir / "img1"
    det_path = seq_dir / "det" / "det.txt"
    gt_path = seq_dir / "gt" / "gt.txt"

    if not video_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {video_dir}")
    if not det_path.exists():
        raise FileNotFoundError(f"Missing detection file: {det_path}")
    args.config = resolve_repo_path(
        args.config,
        [
            PROJECT_ROOT / "sam2/sam2/configs/samurai/sam2.1_hiera_s.yaml",
            PROJECT_ROOT / "sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
            PROJECT_ROOT / "configs/samurai/sam2.1_hiera_s.yaml",
        ],
    )
    args.checkpoint = resolve_repo_path(
        args.checkpoint,
        [
            PROJECT_ROOT / "sam2/checkpoints/sam2.1_hiera_small.pt",
        ],
    )

    if not args.config.exists():
        raise FileNotFoundError(f"Missing config file: {args.config}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {args.checkpoint}")

    output_root = args.output_root / args.sequence / "sam2_samurai_dam"
    packed_mask_dir = output_root / "masks_packed_png"
    per_object_mask_dir = output_root / "masks_per_object"
    video_output_dir = output_root / "videos"
    metrics_dir = output_root / "metrics"
    mot_txt_path = output_root / "tracking_mot.txt"
    for folder in [packed_mask_dir, per_object_mask_dir, video_output_dir, metrics_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not frame_paths:
        raise RuntimeError(f"No frames found in {video_dir}")

    detections = read_mot_boxes(det_path, args.conf_thresh, is_gt=False)
    gt_boxes_by_frame = read_mot_boxes(gt_path, 0.0, is_gt=True) if gt_path.exists() else defaultdict(list)

    predictor = build_predictor(args, device)
    num_frames = len(frame_paths) if args.max_frames <= 0 else min(len(frame_paths), args.max_frames)
    frame_paths = frame_paths[:num_frames]
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    fps = read_fps(seq_dir, args.fps)
    video_path = video_output_dir / f"{args.sequence}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    mot_acc = mm.MOTAccumulator(auto_id=True) if gt_path.exists() else None
    mot_lines = []

    frame_level_stats = []
    per_object_stats = {}
    total_tracks_created = 0
    if args.redetect_interval <= 0:
        raise ValueError("--redetect_interval must be a positive integer")

    next_track_id = 1
    carryover_tracks = {}
    segment_start = 0

    try:
        while segment_start < num_frames:
            segment_end = min(segment_start + args.redetect_interval, num_frames - 1)
            segment_frames = segment_end - segment_start + 1
            frame_number_1_based = segment_start + 1

            frame_dets = sort_detections_for_frame(
                detections.get(frame_number_1_based, []),
                width,
                height,
                limit=args.max_objects if segment_start == 0 else 0,
            )
            active_boxes = {
                track_id: track_state["bbox"]
                for track_id, track_state in carryover_tracks.items()
                if track_state.get("bbox") is not None
            }
            _, _, unmatched_det_indices = greedy_match_boxes(
                active_boxes,
                frame_dets,
                args.birth_iou_threshold,
            )

            segment_seeds = []
            for track_id in sorted(carryover_tracks):
                segment_seeds.append(
                    {
                        "track_id": track_id,
                        "mask": carryover_tracks[track_id].get("mask"),
                        "bbox": carryover_tracks[track_id].get("bbox"),
                    }
                )
            for det_idx in unmatched_det_indices:
                det = frame_dets[det_idx]
                track_id = next_track_id
                next_track_id += 1
                total_tracks_created += 1
                per_object_stats.setdefault(track_id, {"frames_present": 0, "pixel_sum": 0})
                segment_seeds.append({"track_id": track_id, "mask": None, "bbox": det["bbox_xyxy"]})

            continued_count = len(carryover_tracks)
            born_count = len(unmatched_det_indices)
            print(
                f"Segment {segment_start + 1}-{segment_end + 1}: "
                f"continuing={continued_count}, new={born_count}, active_seeds={len(segment_seeds)}"
            )

            if not segment_seeds:
                for frame_idx in range(segment_start, segment_end if segment_end < num_frames - 1 else segment_end + 1):
                    write_frame_artifacts(
                        frame_idx=frame_idx,
                        frame_path=frame_paths[frame_idx],
                        frame_pred_boxes={},
                        frame_pred_masks={},
                        active_objects=0,
                        packed_mask_dir=packed_mask_dir,
                        per_object_mask_dir=per_object_mask_dir,
                        writer=writer,
                        mot_lines=mot_lines,
                        frame_level_stats=frame_level_stats,
                        per_object_stats=per_object_stats,
                        gt_boxes_by_frame=gt_boxes_by_frame,
                        mot_acc=mot_acc,
                        save_per_object_masks=args.save_per_object_masks,
                    )
                    if frame_idx % 25 == 0 or frame_idx == num_frames - 1:
                        print(f"Processed frame {frame_idx + 1}/{num_frames} with 0 active objects")
                carryover_tracks = {}
                segment_start = segment_end if segment_end < num_frames - 1 else num_frames
                continue

            inference_state = prepare_segment_state(predictor, video_dir, segment_start, segment_frames)
            for seed in segment_seeds:
                if seed["mask"] is not None:
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=seed["track_id"],
                        mask=seed["mask"],
                    )
                else:
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=seed["track_id"],
                        box=seed["bbox"],
                    )

            boundary_tracks = {}
            for local_frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=segment_frames,
                reverse=False,
            ):
                frame_idx = segment_start + local_frame_idx
                frame_pred_boxes, frame_pred_masks, active_objects = build_frame_outputs(obj_ids, video_res_masks)
                is_boundary_frame = frame_idx == segment_end and frame_idx < num_frames - 1
                if is_boundary_frame:
                    boundary_tracks = {
                        obj_id: {"mask": frame_pred_masks[obj_id].astype(bool), "bbox": frame_pred_boxes[obj_id]}
                        for obj_id in frame_pred_boxes
                    }
                    continue

                write_frame_artifacts(
                    frame_idx=frame_idx,
                    frame_path=frame_paths[frame_idx],
                    frame_pred_boxes=frame_pred_boxes,
                    frame_pred_masks=frame_pred_masks,
                    active_objects=active_objects,
                    packed_mask_dir=packed_mask_dir,
                    per_object_mask_dir=per_object_mask_dir,
                    writer=writer,
                    mot_lines=mot_lines,
                    frame_level_stats=frame_level_stats,
                    per_object_stats=per_object_stats,
                    gt_boxes_by_frame=gt_boxes_by_frame,
                    mot_acc=mot_acc,
                    save_per_object_masks=args.save_per_object_masks,
                )

                if frame_idx % 25 == 0 or frame_idx == num_frames - 1:
                    print(f"Processed frame {frame_idx + 1}/{num_frames} with {active_objects} active objects")

            carryover_tracks = boundary_tracks
            segment_start = segment_end if segment_end < num_frames - 1 else num_frames
    finally:
        writer.release()

    avg_active_objects = float(np.mean([x["active_objects"] for x in frame_level_stats])) if frame_level_stats else 0.0
    objects_with_any_mask = sum(1 for stats in per_object_stats.values() if stats["frames_present"] > 0)

    serializable_per_object = {}
    for obj_id, stats in per_object_stats.items():
        serializable_per_object[str(obj_id)] = {
            "frames_present": stats["frames_present"],
            "pixel_sum": stats["pixel_sum"],
        }

    mot_metrics = {"status": "gt_missing"}
    if mot_acc is not None:
        mh = mm.metrics.create()
        summary = mh.compute(
            mot_acc,
            metrics=[
                "mota",
                "idf1",
                "precision",
                "recall",
                "num_switches",
                "mostly_tracked",
                "mostly_lost",
                "num_unique_objects",
            ],
            name=args.sequence,
        )
        row = summary.loc[args.sequence]
        num_unique_objects = int(row["num_unique_objects"])
        mostly_tracked = int(row["mostly_tracked"])
        mostly_lost = int(row["mostly_lost"])
        mot_metrics = {
            "status": "success",
            "mota": float(row["mota"]),
            "idf1": float(row["idf1"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "id_switches": int(row["num_switches"]),
            "mostly_tracked": mostly_tracked,
            "mostly_lost": mostly_lost,
            "num_unique_objects": num_unique_objects,
            "mostly_tracked_ratio": float(mostly_tracked / num_unique_objects) if num_unique_objects else 0.0,
            "mostly_lost_ratio": float(mostly_lost / num_unique_objects) if num_unique_objects else 0.0,
            "mota_percent": float(row["mota"] * 100.0),
            "idf1_percent": float(row["idf1"] * 100.0),
            "precision_percent": float(row["precision"] * 100.0),
            "recall_percent": float(row["recall"] * 100.0),
            "mostly_tracked_ratio_percent": float((mostly_tracked / num_unique_objects) * 100.0) if num_unique_objects else 0.0,
            "mostly_lost_ratio_percent": float((mostly_lost / num_unique_objects) * 100.0) if num_unique_objects else 0.0,
        }

    mot_txt_path.write_text("\n".join(mot_lines) + ("\n" if mot_lines else ""))

    metrics = {
        "sequence": args.sequence,
        "device_used": device,
        "frames_processed": num_frames,
        "objects_prompted": total_tracks_created,
        "objects_with_any_mask": objects_with_any_mask,
        "avg_active_objects_per_frame": avg_active_objects,
        "redetect_interval": args.redetect_interval,
        "packed_mask_dir": str(packed_mask_dir),
        "per_object_mask_dir": str(per_object_mask_dir),
        "video_path": str(video_path),
        "mot_txt_path": str(mot_txt_path),
        "mot_metrics": mot_metrics,
        "per_object": serializable_per_object,
        "frame_level_stats": frame_level_stats,
    }

    metrics_json_path = metrics_dir / "tracking_metrics.json"
    summary_txt_path = metrics_dir / "tracking_metrics.txt"
    metrics_json_path.write_text(json.dumps(metrics, indent=2))
    write_summary(summary_txt_path, metrics)

    return metrics


def resolve_device_attempts(requested_device):
    if requested_device == "auto":
        if torch.cuda.is_available():
            return ["cuda", "cpu"]
        return ["cpu"]
    if requested_device.startswith("cuda"):
        return [requested_device, "cpu"] if torch.cuda.is_available() else ["cpu"]
    return [requested_device]


def main():
    args = parse_args()
    attempts = resolve_device_attempts(args.device)
    last_exc = None

    for attempt_index, device in enumerate(attempts, start=1):
        try:
            print(f"\n[{attempt_index}/{len(attempts)}] Running tracking on device={device}")
            metrics = run_sequence(args, device)
            print("\nTracking complete.")
            print(f"Masks: {metrics['packed_mask_dir']}")
            print(f"Per-object masks: {metrics['per_object_mask_dir']}")
            print(f"Video: {metrics['video_path']}")
            print(f"Metrics JSON: {Path(metrics['video_path']).parent.parent / 'metrics' / 'tracking_metrics.json'}")
            return
        except RuntimeError as exc:
            last_exc = exc
            should_retry_on_cpu = device.startswith("cuda") and "cpu" in attempts[attempt_index:] and is_cuda_oom(exc)
            cleanup_after_failure()
            if should_retry_on_cpu:
                print(f"CUDA memory was not enough on {device}. Retrying on CPU.")
                continue
            raise
        except Exception as exc:
            last_exc = exc
            cleanup_after_failure()
            raise

    if last_exc is not None:
        raise last_exc


if __name__ == "__main__":
    main()
