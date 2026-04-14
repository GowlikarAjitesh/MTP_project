import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

# Import the modules you cloned
from samurai.samurai_tracker import SAMURAITracker  # or extract the KF + scoring functions
from d4sm.dam_memory import DAM4SAMMemory          # multi-object version from alanlukezic/d4sm
from sam2mot.trajectory_manager import SAM2MOTTrajectoryManager  # exact class I gave you earlier
from sam2mot.cross_object_interaction import CrossObjectInteraction  # from SAM2MOT repo

class InteractiveHybridSAM2:
    def __init__(self, model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml", checkpoint="checkpoints/sam2.1_hiera_b+.pt"):
        self.predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, checkpoint)
        self.inference_state = None

        # Hybrid components
        self.trajectory_manager = SAM2MOTTrajectoryManager()           # from your previous code / SAM2MOT
        self.cross_interaction = CrossObjectInteraction()              # from SAM2MOT
        self.per_track_memory: Dict[int, DAM4SAMMemory] = {}          # DAM4SAM per obj_id
        self.samurai_trackers: Dict[int, SAMURAITracker] = {}         # SAMURAI Kalman + scoring per obj_id

    def init_video(self, video_path_or_frames):
        self.inference_state = self.predictor.init_state(video_path=video_path_or_frames)
        self.per_track_memory.clear()
        self.samurai_trackers.clear()

    def add_user_prompt(self, frame_idx: int, points=None, box=None, labels=None):
        """User clicks anywhere → instantly creates a new robust track"""
        _, obj_id = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=None,          # auto-assign
            points=points,
            box=box,
            labels=labels
        )
        # Initialize hybrid memory + SAMURAI for this new object
        self.per_track_memory[obj_id] = DAM4SAMMemory()          # RAM + DRM
        self.samurai_trackers[obj_id] = SAMURAITracker()        # Kalman + hybrid scoring
        return obj_id

    def propagate_frame(self, frame_idx: int, frame_shape: Tuple[int, int], detections=None):
        """Main per-frame call — does everything"""
        # 1. Official SAM2 forward pass (with current hybrid memories injected)
        masks, scores, logits, boxes = self.predictor.propagate_in_video(
            self.inference_state, start_frame_idx=frame_idx, max_frames=1
        )

        obj_ids = list(self.per_track_memory.keys())
        if not obj_ids:
            return masks, scores, logits, boxes

        # 2. SAMURAI: better mask selection using Kalman + hybrid scoring
        for oid in obj_ids:
            if oid in masks:
                kf_score = self.samurai_trackers[oid].compute_kf_iou_score(boxes[oid])
                hybrid_score = self.samurai_trackers[oid].hybrid_score(
                    scores[oid], kf_score, logits[oid]
                )
                # Replace SAM2's default mask with best hybrid one
                best_idx = hybrid_score.argmax()
                masks[oid] = masks[oid][best_idx]
                scores[oid] = scores[oid][best_idx]
                logits[oid] = logits[oid][best_idx]

        # 3. DAM4SAM: update per-track memory (RAM + DRM using multi-mask ambiguity)
        for oid in obj_ids:
            self.per_track_memory[oid].update(
                mask=masks[oid],
                score=scores[oid],
                multi_masks=self.predictor.get_multi_masks(oid)  # raw 3 masks from SAM2
            )

        # 4. SAM2MOT: Trajectory Manager + Cross-object Interaction
        new_objs, to_remove, reconstructions = self.trajectory_manager.update(
            frame_idx=frame_idx,
            obj_ids=obj_ids,
            logits=[logits[oid] for oid in obj_ids],
            masks=[masks[oid] for oid in obj_ids],
            tracked_boxes=[boxes[oid] for oid in obj_ids],
            detections=detections,
            frame_shape=frame_shape
        )

        # Cross-object occlusion resolution
        self.cross_interaction.resolve_occlusions(masks, logits, obj_ids)

        # 5. Apply manager decisions
        for _, prompt_box in new_objs:                    # rare detector fallback
            self.add_user_prompt(frame_idx, box=prompt_box)

        for oid, recon_box in reconstructions:            # quality reconstruction
            self.predictor.add_new_box(self.inference_state, frame_idx, oid, box=recon_box)

        for oid in to_remove:
            # Just stop drawing it (SAM2 memory can stay or you can clear it)
            if oid in self.per_track_memory:
                del self.per_track_memory[oid]
                del self.samurai_trackers[oid]

        return masks, scores, logits, boxes