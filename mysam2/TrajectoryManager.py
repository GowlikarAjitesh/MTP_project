import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Any, Optional

class SAM2MOTTrajectoryManager:
    """
    Exact implementation of the Trajectory Manager System from SAM2MOT
    (arXiv:2504.04519). Handles:
      - Object addition (3-stage filtering with detector + Hungarian + M_non overlap)
      - Object removal (logit-based lost state + 25-frame tolerance)
      - Quality reconstruction (pending state + high-conf detection match → update keyframe prompt)

    This is a drop-in manager you can use on top of SAM2VideoPredictor.
    No training required. Works with any detector that outputs boxes + scores.

    Hyperparameters are taken directly from the paper (default values).
    """

    def __init__(
        self,
        conf_thresh: float = 0.5,      # 0.5 for Co-DINO-L, 0.4 for Grounding-DINO-L
        overlap_ratio_r: float = 0.7,
        lost_tolerance_frames: int = 25,
        tau_r: float = 8.0,
        tau_p: float = 6.0,
        tau_s: float = 2.0,
    ):
        self.conf_thresh = conf_thresh
        self.r = overlap_ratio_r
        self.lost_tolerance = lost_tolerance_frames
        self.tau_r = tau_r
        self.tau_p = tau_p
        self.tau_s = tau_s

        # Internal state per object (obj_id -> dict)
        # SAM2 already manages memory per object_id; we only store extra bookkeeping
        self.track_states: Dict[int, Dict[str, Any]] = {}

    def _get_object_state(self, logits: float) -> str:
        """Equation (3) from SAM2MOT paper"""
        if logits > self.tau_r:
            return "reliable"
        elif self.tau_p < logits <= self.tau_r:
            return "pending"
        elif self.tau_s < logits <= self.tau_p:
            return "suspicious"
        else:
            return "lost"

    @staticmethod
    def _compute_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """IoU matrix between two sets of boxes [x1, y1, x2, y2]"""
        # boxes1: (N1, 4), boxes2: (N2, 4)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Intersection
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.maximum(rb - lt, 0.0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-6)
        return iou

    def update(
        self,
        frame_idx: int,
        obj_ids: List[int],
        logits: List[float],
        masks: List[np.ndarray],           # list of (H, W) binary masks (one per obj_id)
        tracked_boxes: List[np.ndarray],   # list of [x1, y1, x2, y2] (one per obj_id)
        detections: np.ndarray,            # (N, 5) array: [x1, y1, x2, y2, score]
        frame_shape: Tuple[int, int],      # (H, W) of current frame (for M_non)
    ) -> Tuple[
        List[Tuple[int, np.ndarray]],      # new_objects: (new_obj_id or None, prompt_box)
        List[int],                         # to_remove: obj_ids to deactivate
        List[Tuple[int, np.ndarray]]       # reconstructions: (obj_id, new_keyframe_box)
    ]:
        """
        Call this every frame (or every N frames to save compute).

        Returns three things you act on with your SAM2VideoPredictor:
          1. new_objects → for each, call predictor.add_new_box(...) and get the new obj_id
          2. to_remove → ignore/deactivate these obj_ids in visualization (SAM2 memory stays until you reset)
          3. reconstructions → for each, re-prompt the existing object with the box:
                predictor.add_new_box(obj_id=oid, box=box)  # this acts as keyframe update
        """
        if not obj_ids:
            obj_ids = []
            logits = []
            masks = []
            tracked_boxes = []

        # 1. Update per-object states (removal + pending tracking)
        for oid, log_val, mask_arr, tbox in zip(obj_ids, logits, masks, tracked_boxes):
            if oid not in self.track_states:
                self.track_states[oid] = {
                    "logits_history": [],
                    "lost_counter": 0,
                    "state": "reliable",
                }

            state_dict = self.track_states[oid]
            state_dict["logits_history"].append(log_val)
            if len(state_dict["logits_history"]) > 30:  # keep reasonable history
                state_dict["logits_history"] = state_dict["logits_history"][-30:]

            current_state = self._get_object_state(log_val)
            state_dict["state"] = current_state

            if current_state == "lost":
                state_dict["lost_counter"] += 1
            else:
                state_dict["lost_counter"] = 0

        # 2. Object removal
        to_remove = [
            oid for oid, s in self.track_states.items()
            if s["state"] == "lost" and s["lost_counter"] > self.lost_tolerance
        ]
        for oid in to_remove:
            self.track_states.pop(oid, None)

        # 3. Object addition + quality reconstruction
        new_objects: List[Tuple[int, np.ndarray]] = []
        reconstructions: List[Tuple[int, np.ndarray]] = []

        if len(detections) == 0:
            return new_objects, to_remove, reconstructions

        # Filter high-confidence detections (stage 1)
        high_conf_mask = detections[:, 4] > self.conf_thresh
        high_conf_dets = detections[high_conf_mask]  # (K, 5)

        if len(high_conf_dets) == 0:
            return new_objects, to_remove, reconstructions

        # Get current tracked boxes for Hungarian matching (stage 2)
        tb_array = np.array(tracked_boxes) if tracked_boxes else np.empty((0, 4))
        if len(tb_array) > 0 and len(high_conf_dets) > 0:
            iou_mat = self._compute_box_iou(high_conf_dets[:, :4], tb_array)
            row_ind, col_ind = linear_sum_assignment(-iou_mat)  # maximize IoU
            matched_rows = set(row_ind)
        else:
            matched_rows = set()

        # Candidates for NEW objects = unmatched high-conf detections
        candidate_dets = [high_conf_dets[i] for i in range(len(high_conf_dets)) if i not in matched_rows]

        # Compute M_non = I - union of all tracked masks (stage 3)
        if masks:
            union_mask = np.zeros(frame_shape, dtype=bool)
            for m in masks:
                union_mask |= m.astype(bool)
            M_non = ~union_mask
        else:
            M_non = np.ones(frame_shape, dtype=bool)

        # Check overlap with M_non
        for cand in candidate_dets:
            box = cand[:4].astype(int)
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                continue
            overlap_pixels = np.sum(M_non[y1:y2, x1:x2])
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 0 and (overlap_pixels / box_area) > self.r:
                # Valid new object → caller must call add_new_box and assign obj_id
                new_objects.append((None, box))  # None = new id will be returned by SAM2

        # Quality reconstruction for pending objects (uses the matched pairs from Hungarian)
        if len(tb_array) > 0 and len(high_conf_dets) > 0:
            for r_idx, c_idx in zip(row_ind, col_ind):
                oid = obj_ids[c_idx]
                if oid in self.track_states and self.track_states[oid]["state"] == "pending":
                    recon_box = high_conf_dets[r_idx][:4].astype(int)
                    reconstructions.append((oid, recon_box))
                    # Caller will do: predictor.add_new_box(obj_id=oid, box=recon_box)
                    # This effectively updates the keyframe information in SAM2 memory

        return new_objects, to_remove, reconstructions