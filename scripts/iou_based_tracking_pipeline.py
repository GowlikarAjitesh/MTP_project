import argparse
import configparser
import json
import os
from dataclasses import dataclass

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Compatibility fix for older motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda array_like, dtype=float: np.asarray(array_like, dtype=dtype)

mm.lap.default_solver = "scipy"

PEDESTRIAN_CLASS_ID = 1


@dataclass
class Detection:
    bbox: np.ndarray
    score: float


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    hits: int = 1
    time_since_update: int = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simple MOT tracker on MOT17 train split with video output."
    )
    parser.add_argument(
        "--mot-root",
        default="/home/cs24m118/datasets/MOT17/train",
        help="Path to MOT17 train sequences.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/cs24m118/phase2/outputs/mysam2",
        help="Directory for tracker outputs, videos, and metrics.",
    )
    parser.add_argument(
        "--seq",
        nargs="*",
        help="Optional sequence names to run. Defaults to all sequences.",
    )
    parser.add_argument(
        "--det-score-thresh",
        type=float,
        default=0.5,
        help="Minimum detection confidence to keep.",
    )
    parser.add_argument(
        "--match-iou-thresh",
        type=float,
        default=0.3,
        help="Minimum IoU for associating a detection to an active track.",
    )
    parser.add_argument(
        "--eval-iou-thresh",
        type=float,
        default=0.5,
        help="IoU threshold used during MOT metric computation.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="How many missed frames a track survives before removal.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Track confirmation threshold.",
    )
    return parser.parse_args()


def load_seqinfo(seq_path):
    seqinfo = configparser.ConfigParser()
    seqinfo.read(os.path.join(seq_path, "seqinfo.ini"))
    sequence = seqinfo["Sequence"]
    return {
        "name": sequence["name"],
        "im_dir": sequence["imDir"],
        "frame_rate": int(sequence["frameRate"]),
        "seq_length": int(sequence["seqLength"]),
        "width": int(sequence["imWidth"]),
        "height": int(sequence["imHeight"]),
        "ext": sequence["imExt"],
    }


def load_detections(det_path, score_thresh):
    detections_by_frame = {}
    with open(det_path, "r") as handle:
        for line in handle:
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
    with open(gt_path, "r") as handle:
        for line in handle:
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
                {
                    "id": gt_id,
                    "bbox": np.array([x, y, w, h], dtype=np.float32),
                }
            )
    return gt_by_frame


def bbox_iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def associate_detections_to_tracks(tracks, detections, match_iou_thresh):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            iou_matrix[t_idx, d_idx] = bbox_iou(track.bbox, det.bbox)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

    matches = []
    matched_t = set()
    matched_d = set()

    for t_idx, d_idx in zip(row_ind, col_ind):
        if iou_matrix[t_idx, d_idx] >= match_iou_thresh:
            matches.append((t_idx, d_idx))
            matched_t.add(t_idx)
            matched_d.add(d_idx)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]

    return matches, unmatched_tracks, unmatched_dets


def draw_tracks(image, active_tracks):
    canvas = image.copy()
    for track in active_tracks:
        x, y, w, h = track.bbox.astype(int)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"ID {track.track_id}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def run_tracker_on_sequence(seq_name, seq_path, args, results_dir, videos_dir):
    seqinfo = load_seqinfo(seq_path)
    detections = load_detections(
        os.path.join(seq_path, "det", "det.txt"), args.det_score_thresh
    )
    gt_by_frame = load_ground_truth(os.path.join(seq_path, "gt", "gt.txt"))

    image_dir = os.path.join(seq_path, seqinfo["im_dir"])
    active_tracks = []
    next_track_id = 1
    result_rows = []

    # Always create video
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{seq_name}.avi")
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"XVID"),   # More reliable than MJPG
        seqinfo["frame_rate"],
        (seqinfo["width"], seqinfo["height"]),
    )

    print(f"  → Processing {seq_name} ({seqinfo['seq_length']} frames)")

    for frame_id in range(1, seqinfo["seq_length"] + 1):
        frame_detections = detections.get(frame_id, [])

        # Age tracks
        for track in active_tracks:
            track.time_since_update += 1

        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            active_tracks, frame_detections, args.match_iou_thresh
        )

        # Update matched tracks
        for t_idx, d_idx in matches:
            det = frame_detections[d_idx]
            track = active_tracks[t_idx]
            track.bbox = det.bbox.copy()
            track.score = det.score
            track.hits += 1
            track.time_since_update = 0

        # Create new tracks
        for d_idx in unmatched_dets:
            det = frame_detections[d_idx]
            active_tracks.append(
                Track(next_track_id, det.bbox.copy(), det.score)
            )
            next_track_id += 1

        # Remove old tracks
        active_tracks = [t for t in active_tracks if t.time_since_update <= args.max_age]

        # Save output for confirmed tracks
        for track in active_tracks:
            if track.time_since_update == 0 and (track.hits >= args.min_hits or frame_id <= args.min_hits):
                x, y, w, h = track.bbox.tolist()
                result_rows.append([
                    frame_id, track.track_id, round(x, 2), round(y, 2),
                    round(w, 2), round(h, 2), round(track.score, 4), -1, -1, -1
                ])

        # Write video frame
        image_path = os.path.join(image_dir, f"{frame_id:06d}{seqinfo['ext']}")
        image = cv2.imread(image_path)
        if image is not None:
            visible_tracks = [t for t in active_tracks if t.time_since_update == 0]
            drawn = draw_tracks(image, visible_tracks)
            video_writer.write(drawn)
        else:
            print(f"    Warning: Could not read {image_path}")

    video_writer.release()

    # Save tracker results
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{seq_name}.txt")
    with open(result_path, "w") as f:
        for row in result_rows:
            f.write(",".join(map(str, row)) + "\n")

    # Evaluate sequence
    metrics = evaluate_sequence(gt_by_frame, result_rows, args.eval_iou_thresh)
    metrics.update({
        "num_output_rows": len(result_rows),
        "result_path": result_path,
        "video_path": video_path
    })

    return metrics


def evaluate_sequence(gt_by_frame, result_rows, eval_iou_thresh):
    pred_by_frame = {}
    for row in result_rows:
        frame_id, track_id, x, y, w, h = row[:6]
        pred_by_frame.setdefault(frame_id, []).append({
            "id": int(track_id),
            "bbox": np.array([x, y, w, h], dtype=np.float32)
        })

    accumulator = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    for frame_id in all_frames:
        gt_items = gt_by_frame.get(frame_id, [])
        pred_items = pred_by_frame.get(frame_id, [])

        gt_ids = [item["id"] for item in gt_items]
        pred_ids = [item["id"] for item in pred_items]
        gt_boxes = [item["bbox"] for item in gt_items]
        pred_boxes = [item["bbox"] for item in pred_items]

        if gt_boxes and pred_boxes:
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0 - eval_iou_thresh)
        else:
            distances = np.empty((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

        accumulator.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=mm.metrics.motchallenge_metrics, name="sequence")
    metrics = summary.iloc[0].to_dict()

    for k, v in list(metrics.items()):
        if isinstance(v, (np.floating, np.integer)):
            metrics[k] = float(v)

    return metrics


def evaluate_all_sequences(mot_root, results_dir, seq_names, eval_iou_thresh):
    accumulators = []
    names = []

    for seq_name in seq_names:
        gt_path = os.path.join(mot_root, seq_name, "gt", "gt.txt")
        gt_by_frame = load_ground_truth(gt_path)

        result_path = os.path.join(results_dir, f"{seq_name}.txt")
        result_rows = []

        if os.path.exists(result_path):
            with open(result_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    result_rows.append([int(float(p)) if i < 2 else float(p) for i, p in enumerate(parts[:6])])

        # Build predictions
        pred_by_frame = {}
        for row in result_rows:
            frame_id, track_id, x, y, w, h = row
            pred_by_frame.setdefault(frame_id, []).append({
                "id": int(track_id), "bbox": np.array([x, y, w, h], dtype=np.float32)
            })

        accumulator = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

        for frame_id in all_frames:
            gt_items = gt_by_frame.get(frame_id, [])
            pred_items = pred_by_frame.get(frame_id, [])

            distances = mm.distances.iou_matrix(
                [item["bbox"] for item in gt_items],
                [item["bbox"] for item in pred_items],
                max_iou=1.0 - eval_iou_thresh
            ) if gt_items and pred_items else np.empty((len(gt_items), len(pred_items)), dtype=np.float32)

            accumulator.update(
                [item["id"] for item in gt_items],
                [item["id"] for item in pred_items],
                distances
            )

        accumulators.append(accumulator)
        names.append(seq_name)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True
    )

    return summary


def main():
    args = parse_args()

    output_root = os.path.abspath(args.output_root)
    results_dir = os.path.join(output_root, "mot_results")
    videos_dir = os.path.join(output_root, "videos")
    metrics_json_path = os.path.join(output_root, "metrics.json")
    metrics_csv_path = os.path.join(output_root, "metrics_summary.csv")

    if args.seq:
        seq_names = args.seq
    else:
        seq_names = sorted(
            entry for entry in os.listdir(args.mot_root)
            if os.path.isdir(os.path.join(args.mot_root, entry))
        )

    per_sequence_metrics = {}
    for seq_name in seq_names:
        seq_path = os.path.join(args.mot_root, seq_name)
        per_sequence_metrics[seq_name] = run_tracker_on_sequence(
            seq_name, seq_path, args, results_dir, videos_dir
        )

    print("\nComputing overall metrics...")
    summary_df = evaluate_all_sequences(
        args.mot_root, results_dir, seq_names, args.eval_iou_thresh
    )

    # Save summary as CSV
    os.makedirs(output_root, exist_ok=True)
    summary_df.to_csv(metrics_csv_path)

    # Save full details as JSON
    metrics_payload = {
        "settings": vars(args),
        "per_sequence": per_sequence_metrics,
        "summary": summary_df.reset_index().to_dict(orient="records"),
    }

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 90)
    print(summary_df.to_string())
    print("=" * 90)
    print(f"✅ Videos saved in: {videos_dir}")
    print(f"✅ Tracker outputs saved in: {results_dir}")
    print(f"✅ Metrics summary saved as CSV: {metrics_csv_path}")
    print(f"✅ Full metrics saved as JSON: {metrics_json_path}")


if __name__ == "__main__":
    main()