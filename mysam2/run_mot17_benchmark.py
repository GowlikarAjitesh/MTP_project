#!/usr/bin/env python3
"""
BERT Tracker MOT17 Benchmarking Script
Runs BERT tracker on all MOT17 sequences and evaluates performance.
Saves: output masks, videos, and metrics.
"""

import os
import sys
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging
from configparser import ConfigParser
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import BERT tracker
try:
    from bert_tracker import BERTTracker
except ImportError:
    logger.error("bert_tracker module not found!")
    sys.exit(1)


class MOT17Benchmark:
    """MOT17 Benchmark Runner"""
    
    def __init__(
        self,
        dataset_root="/home/cs24m118/datasets/MOT17",
        output_root="/home/cs24m118/phase2/mysam2/mot17_results",
        model_cfg="sam2_hiera_l",
        checkpoint=None,
        device="cuda:0",
        skip_video=False,
        skip_masks=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self.skip_video = skip_video
        self.skip_masks = skip_masks
        
        # Create output directory structure
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "masks").mkdir(exist_ok=True)
        (self.output_root / "videos").mkdir(exist_ok=True)
        (self.output_root / "metrics").mkdir(exist_ok=True)
        (self.output_root / "tracks").mkdir(exist_ok=True)
        
        # Load checkpoint if not provided
        if checkpoint is None:
            self.checkpoint = self._find_checkpoint()
        if self.checkpoint is None:
            raise FileNotFoundError(
                "No SAM2 checkpoint found. Pass --checkpoint explicitly or place weights in one of the searched directories."
            )
        
        logger.info(f"Output directory: {self.output_root}")
        logger.info(f"Using checkpoint: {self.checkpoint}")

    def _find_checkpoint(self):
        """Find SAM2 checkpoint"""
        possible_paths = [
            "/home/cs24m118/sam2_backup/checkpoints",
            "/home/cs24m118/phase2/sam2/checkpoints",
            os.path.expanduser("~/.cache/sam2"),
        ]
        
        # Map model configs to file names
        checkpoint_names = {
            "sam2_hiera_l": ["sam2_hiera_l.pt", "sam2_hiera_large.pt"],
            "sam2_hiera_b": ["sam2_hiera_b.pt", "sam2_hiera_base.pt"],
            "sam2_hiera_s": ["sam2_hiera_s.pt", "sam2_hiera_small.pt"],
            "sam2_hiera_t": ["sam2_hiera_t.pt", "sam2_hiera_tiny.pt"],
        }
        
        names_to_try = checkpoint_names.get(self.model_cfg, [f"{self.model_cfg}.pt"])
        
        for path in possible_paths:
            for checkpoint_name in names_to_try:
                checkpoint_path = Path(path) / checkpoint_name
                if checkpoint_path.exists():
                    logger.info(f"Found checkpoint: {checkpoint_path}")
                    return str(checkpoint_path)
        
        logger.warning(f"Checkpoint not found for {self.model_cfg}")
        logger.info(f"Tried: {names_to_try}")
        logger.info(f"Locations: {possible_paths}")
        return None

    def get_sequences(self, split="train"):
        """Get all MOT17 sequences"""
        split_dir = self.dataset_root / split
        if not split_dir.exists():
            logger.warning(f"Dataset split not found: {split_dir}")
            return []
        
        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(sequences)} sequences in {split}")
        return sequences

    def load_detections(self, sequence_dir):
        """Load detector outputs (MOT17 format)"""
        det_file = sequence_dir / "det" / "det.txt"
        if not det_file.exists():
            logger.warning(f"Detections not found: {det_file}")
            return defaultdict(list)
        
        detections = defaultdict(list)
        with open(det_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    frame_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[2:6])
                    conf = float(parts[6])
                    # Keep in [x1, y1, x2, y2, conf] format for consistency
                    detections[frame_id].append(np.array([x, y, x+w, y+h, conf]))
        
        return detections

    def load_sequence_info(self, sequence_dir):
        """Load sequence information from seqinfo.ini"""
        info_file = sequence_dir / "seqinfo.ini"
        info = {
            'imDir': 'img1',
            'frameRate': 30,
            'seqLength': None,
            'imWidth': 1920,
            'imHeight': 1080,
        }
        
        if info_file.exists():
            config = ConfigParser()
            config.read(info_file)
            if 'Sequence' in config:
                seq_cfg = config['Sequence']
                info['imDir'] = seq_cfg.get('imDir', 'img1')
                info['frameRate'] = int(seq_cfg.get('frameRate', 30))
                info['seqLength'] = int(seq_cfg.get('seqLength', 0))
                info['imWidth'] = int(seq_cfg.get('imWidth', 1920))
                info['imHeight'] = int(seq_cfg.get('imHeight', 1080))
        
        return info

    def load_frame_images(self, sequence_dir, seq_info):
        """Load all frame images from sequence"""
        img_dir = sequence_dir / seq_info['imDir']
        frames = []
        
        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            return frames
        
        img_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        for img_file in img_files[:seq_info['seqLength']] if seq_info['seqLength'] else img_files:
            frame = cv2.imread(str(img_file))
            if frame is None:
                logger.warning(f"Failed to read frame: {img_file}")
                continue
            frames.append(frame)
        
        logger.info(f"Loaded {len(frames)} frames from {sequence_dir.name}")
        return frames

    def track_sequence(self, sequence_dir, tracker, save_masks=True, save_video=True):
        """Track single sequence"""
        seq_name = sequence_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sequence: {seq_name}")
        logger.info(f"{'='*60}")
        
        # Load sequence info and frames
        seq_info = self.load_sequence_info(sequence_dir)
        frames = self.load_frame_images(sequence_dir, seq_info)
        detections = self.load_detections(sequence_dir)
        
        if not frames:
            logger.error(f"No frames found for {seq_name}")
            return None
        
        frame_height, frame_width = frames[0].shape[:2]
        logger.info(f"Sequence: {len(frames)} frames, {frame_width}x{frame_height}")
        
        # Reset tracker
        tracker.reset()
        
        # Tracking results storage
        track_results = {
            'sequence': seq_name,
            'total_frames': len(frames),
            'frame_detections': {},
            'object_trajectories': defaultdict(list),
        }
        
        all_masks = {}
        frame_masks = []
        
        # Video writer setup
        video_writer = None
        if save_video and not self.skip_video:
            video_path = self.output_root / "videos" / f"{seq_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, seq_info['frameRate'],
                (frame_width, frame_height)
            )
            logger.info(f"Video output: {video_path}")
        
        try:
            # Initialize with first frame detections
            frame_det = detections.get(1, [])
            active_tracker_ids = []
            det_to_track = {}  # Map detection index to tracker object id
            init_result = None
            
            if len(frame_det) > 0:
                logger.info(f"Initializing with {len(frame_det)} detections")
                det_array = np.asarray(frame_det, dtype=np.float32)
                if hasattr(tracker, "initialize_from_detections"):
                    initialized_ids = tracker.initialize_from_detections(frames[0], det_array)
                    if initialized_ids:
                        logger.info(f"Initialized objects: {initialized_ids}")
                        active_tracker_ids.extend(initialized_ids)
                        if 0 in tracker.current_masks:
                            init_result = {
                                "pred_mask": tracker.current_masks[0],
                                "obj_id": 0,
                            }
                if not active_tracker_ids:
                    best_det_idx = np.argmax([d[4] for d in frame_det])
                    best_det = frame_det[best_det_idx]
                    bbox = [best_det[0], best_det[1], best_det[2]-best_det[0], best_det[3]-best_det[1]]
                    init_result = tracker.initialize(frames[0], bbox=bbox)
                    if init_result['obj_id'] is not None:
                        logger.info(f"Initialized main object {init_result['obj_id']}")
                        active_tracker_ids.append(init_result['obj_id'])
                        det_to_track[best_det_idx] = init_result['obj_id']
            else:
                logger.warning("No detections in first frame")
                h, w = frames[0].shape[:2]
                init_result = tracker.initialize(frames[0], bbox=[0, 0, w, h])
                if init_result['obj_id'] is not None:
                    active_tracker_ids.append(init_result['obj_id'])

            if not active_tracker_ids:
                logger.error("Tracker initialization produced no active objects for %s", seq_name)
                return None
            
            # Store frame 1 results
            if len(frame_det) > 0:
                track_results['frame_detections'][1] = len(frame_det)
                track_results['object_trajectories'][0].append({
                    'frame': 0,
                    'detections': len(frame_det),
                    'confidence': float(np.mean([d[4] for d in frame_det]))
                })

            if save_video and not self.skip_video and video_writer:
                init_mask = init_result.get('pred_mask') if isinstance(init_result, dict) else None
                if init_mask is not None:
                    vis_frame = self._visualize_frame(frames[0], init_mask, frame_det)
                else:
                    vis_frame = self._visualize_detections(frames[0], frame_det)
                video_writer.write(vis_frame)

            if isinstance(init_result, dict) and init_result.get('pred_mask') is not None:
                all_masks[0] = init_result['pred_mask']
            
            # Track remaining frames
            for frame_idx in tqdm(range(1, len(frames)), desc=f"Tracking {seq_name}"):
                frame = frames[frame_idx]
                frame_det = detections.get(frame_idx + 1, [])
                
                # Track with current model
                det_array = np.asarray(frame_det, dtype=np.float32) if len(frame_det) > 0 else None
                track_result = tracker.track(frame, detections=det_array, return_all=True)
                pred_mask = track_result.get('pred_mask')
                if frame_idx < 5:
                    logger.info(
                        "Frame %s: obj_ids=%s mask_present=%s nonzero=%s",
                        frame_idx,
                        track_result.get("obj_ids", []),
                        pred_mask is not None,
                        int(np.sum(pred_mask > 0)) if pred_mask is not None else 0,
                    )
                
                if pred_mask is not None and np.any(pred_mask > 0):
                    all_masks[frame_idx] = pred_mask
                    frame_masks.append(pred_mask)
                    
                    # Store trajectory
                    avg_conf = float(np.mean([d[4] for d in frame_det])) if len(frame_det) > 0 else 0
                    track_results['object_trajectories'][0].append({
                        'frame': frame_idx,
                        'detections': len(frame_det),
                        'confidence': avg_conf,
                        'mask_pixels': int(np.sum(pred_mask > 0))
                    })
                    
                    # Visualize with mask overlay and detection boxes
                    if save_video and not self.skip_video and video_writer:
                        vis_frame = self._visualize_frame(frame, pred_mask, frame_det)
                        video_writer.write(vis_frame)
                else:
                    if save_video and not self.skip_video and video_writer:
                        vis_frame = self._visualize_detections(frame, frame_det)
                        video_writer.write(vis_frame)
                
                track_results['frame_detections'][frame_idx + 1] = len(frame_det)
            
            # Release video writer
            if video_writer:
                video_writer.release()
                logger.info(f"Video saved successfully")
            
            # Save masks
            if save_masks and not self.skip_masks:
                self._save_masks(seq_name, all_masks)
            
            # Save trajectory data
            track_file = self.output_root / "tracks" / f"{seq_name}_tracks.json"
            with open(track_file, 'w') as f:
                json.dump(track_results, f, indent=2, default=str)
            
            # Calculate metrics
            metrics = self._calculate_metrics(all_masks, detections, len(frames))
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error tracking {seq_name}: {e}", exc_info=True)
            if video_writer:
                video_writer.release()
            return None

    def _visualize_frame(self, frame, mask, detections, thickness=2):
        """Visualize frame with tracking results"""
        vis_frame = frame.copy()
        
        # Draw mask overlay (green)
        if mask is not None and np.any(mask > 0):
            mask_overlay = vis_frame.copy()
            mask_overlay[mask > 0] = [0, 255, 0]  # Green for tracked mask
            vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_overlay, 0.3, 0)
        
        # Draw detections as blue boxes
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4] if len(det) > 4 else 0
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(vis_frame.shape[1], x2), min(vis_frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                # Draw detection box in blue
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
                # Draw confidence score
                text = f"Det: {conf:.2f}"
                cv2.putText(vis_frame, text, (x1, max(15, y1-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return vis_frame

    def _visualize_detections(self, frame, detections, thickness=2):
        """Visualize only detections (when mask not available)"""
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4] if len(det) > 4 else 0
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(vis_frame.shape[1], x2), min(vis_frame.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
                text = f"Det: {conf:.2f}"
                cv2.putText(vis_frame, text, (x1, max(15, y1-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return vis_frame

    def _save_masks(self, seq_name, masks_dict):
        """Save masks to disk"""
        mask_dir = self.output_root / "masks" / seq_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_idx, mask in masks_dict.items():
            mask_path = mask_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(mask_path), mask * 255)
        
        logger.info(f"Saved {len(masks_dict)} masks for {seq_name}")

    def _calculate_metrics(self, masks, detections, total_frames):
        """Calculate tracking metrics"""
        total_det_count = sum(len(v) for v in detections.values())
        
        metrics = {
            'total_frames': total_frames,
            'tracked_frames': len(masks),
            'total_detections': total_det_count,
            'frames_with_detections': len([v for v in detections.values() if len(v) > 0]),
        }
        
        # Average detection confidence
        all_confidences = []
        for v in detections.values():
            for det in v:
                if isinstance(det, np.ndarray) and len(det) > 4:
                    all_confidences.append(float(det[4]))
                elif isinstance(det, (list, tuple)) and len(det) > 4:
                    all_confidences.append(float(det[4]))
        
        metrics['average_confidence'] = float(np.mean(all_confidences)) if all_confidences else 0.0
        metrics['min_confidence'] = float(np.min(all_confidences)) if all_confidences else 0.0
        metrics['max_confidence'] = float(np.max(all_confidences)) if all_confidences else 0.0
        
        # Mask coverage
        if masks:
            coverage_ratios = []
            for mask in masks.values():
                if mask is not None:
                    coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                    coverage_ratios.append(coverage)
            metrics['average_mask_coverage'] = float(np.mean(coverage_ratios)) if coverage_ratios else 0.0
            metrics['total_mask_pixels'] = int(sum(np.sum(m > 0) for m in masks.values() if m is not None))
        
        # Detections per frame statistics
        det_counts = [len(v) for v in detections.values()]
        metrics['average_detections_per_frame'] = float(np.mean(det_counts)) if det_counts else 0.0
        metrics['max_detections_in_frame'] = int(np.max(det_counts)) if det_counts else 0
        
        return metrics

    def run_benchmark(self, split="train", sequences=None):
        """Run benchmark on MOT17"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting MOT17 Benchmark on {split} split")
        logger.info(f"{'='*60}")
        
        torch.cuda.empty_cache()
        
        # Initialize tracker
        logger.info(f"Initializing BERT Tracker...")
        tracker = BERTTracker(
            model_cfg=self.model_cfg,
            checkpoint=self.checkpoint,
            device=self.device,
            use_drm=True,
            drm_interval=5,
        )
        if getattr(tracker, "predictor", None) is None:
            raise RuntimeError("BERTTracker predictor failed to initialize.")
        
        # Get sequences
        all_sequences = self.get_sequences(split)
        if sequences:
            all_sequences = [s for s in all_sequences if any(seq in s.name for seq in sequences)]
        
        logger.info(f"Will process {len(all_sequences)} sequences")
        
        all_metrics = {}
        failed_sequences = []
        
        # Process each sequence
        for sequence_dir in all_sequences:
            try:
                seq_name = sequence_dir.name
                metrics = self.track_sequence(
                    sequence_dir,
                    tracker,
                    save_masks=not self.skip_masks,
                    save_video=not self.skip_video
                )
                
                if metrics:
                    all_metrics[seq_name] = metrics
                    logger.info(f"✓ Completed {seq_name}: {metrics['tracked_frames']}/{metrics['total_frames']} frames")
                else:
                    failed_sequences.append(seq_name)
                    logger.error(f"✗ Failed {seq_name}")
            
            except Exception as e:
                logger.error(f"✗ Error processing {sequence_dir.name}: {e}", exc_info=True)
                failed_sequences.append(sequence_dir.name)
        
        # Save aggregated metrics
        self._save_aggregate_metrics(all_metrics, failed_sequences)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark Completed")
        logger.info(f"Processed: {len(all_metrics)} sequences")
        logger.info(f"Failed: {len(failed_sequences)}")
        logger.info(f"Results saved to: {self.output_root}")
        logger.info(f"{'='*60}\n")
        
        return all_metrics

    def _save_aggregate_metrics(self, metrics_dict, failed_sequences):
        """Save aggregate metrics to JSON"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_sequences': len(metrics_dict) + len(failed_sequences),
            'successful_sequences': len(metrics_dict),
            'failed_sequences': len(failed_sequences),
            'failed_names': failed_sequences,
            'sequences': metrics_dict,
        }
        
        # Add aggregate statistics
        if metrics_dict:
            total_frames = sum(m['total_frames'] for m in metrics_dict.values())
            tracked_frames = sum(m['tracked_frames'] for m in metrics_dict.values())
            total_dets = sum(m['total_detections'] for m in metrics_dict.values())
            
            summary['aggregate'] = {
                'total_frames': total_frames,
                'tracked_frames': tracked_frames,
                'tracking_ratio': tracked_frames / total_frames if total_frames > 0 else 0,
                'total_detections': total_dets,
                'average_detections_per_frame': float(np.mean([
                    m.get('average_detections_per_frame', 0) for m in metrics_dict.values()
                ])),
                'average_detection_confidence': float(np.mean([
                    m.get('average_confidence', 0) for m in metrics_dict.values()
                ])),
                'average_mask_coverage': float(np.mean([
                    m.get('average_mask_coverage', 0) for m in metrics_dict.values()
                ])),
                'total_mask_pixels': int(sum(m.get('total_mask_pixels', 0) for m in metrics_dict.values())),
            }
        
        metrics_file = self.output_root / "metrics" / "summary.json"
        with open(metrics_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Print summary
        if 'aggregate' in summary:
            logger.info("\n" + "="*60)
            logger.info("AGGREGATE BENCHMARKING METRICS")
            logger.info("="*60)
            agg = summary['aggregate']
            logger.info(f"Total Frames: {agg['total_frames']}")
            logger.info(f"Tracked Frames: {agg['tracked_frames']}/{agg['total_frames']} ({agg['tracking_ratio']*100:.1f}%)")
            logger.info(f"Total Detections: {agg['total_detections']}")
            logger.info(f"Avg Detections/Frame: {agg['average_detections_per_frame']:.2f}")
            logger.info(f"Avg Detection Confidence: {agg['average_detection_confidence']:.4f}")
            logger.info(f"Avg Mask Coverage: {agg['average_mask_coverage']:.4f}")
            logger.info("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BERT Tracker MOT17 Benchmark")
    parser.add_argument('--dataset-root', default="/home/cs24m118/datasets/MOT17",
                       help="MOT17 dataset root")
    parser.add_argument('--output-root', default="/home/cs24m118/phase2/mysam2/mot17_results",
                       help="Output directory")
    parser.add_argument('--model', default="sam2_hiera_l",
                       help="SAM2 model config")
    parser.add_argument('--checkpoint', default=None,
                       help="Model checkpoint path")
    parser.add_argument('--device', default="cuda:0",
                       help="GPU device")
    parser.add_argument('--split', default="train",
                       help="Dataset split (train/test)")
    parser.add_argument('--sequences', nargs='+', default=None,
                       help="Specific sequences to run")
    parser.add_argument('--skip-video', action='store_true',
                       help="Skip video generation")
    parser.add_argument('--skip-masks', action='store_true',
                       help="Skip mask saving")
    
    args = parser.parse_args()
    
    benchmark = MOT17Benchmark(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        model_cfg=args.model,
        checkpoint=args.checkpoint,
        device=args.device,
        skip_video=args.skip_video,
        skip_masks=args.skip_masks,
    )
    
    metrics = benchmark.run_benchmark(
        split=args.split,
        sequences=args.sequences
    )
    
    return metrics


if __name__ == "__main__":
    main()
