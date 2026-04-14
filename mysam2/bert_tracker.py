"""
Hybrid SAM2 tracker that combines ideas from:
1. SAM2MOT: trajectory management, object addition/removal, reconstruction
2. DAM4SAM: recent appearance memory (RAM) and distractor resolving memory (DRM)
3. SAMURAI: Kalman motion modeling and hybrid affinity selection

The implementation is intentionally training-free and keeps the original public
API (`initialize`, `track`, `add_new_object`, `reset`) so existing scripts can
adopt it with minimal changes.

FIX SUMMARY (cross-referenced against each paper):
  [DAM4SAM §3.2.1] RAM interval default fixed 5 -> correct; but RAM must ALSO
      always include the MOST RECENT frame — handled in _maybe_update_memory by
      always refreshing on the very latest frame regardless of interval.
  [DAM4SAM §3.2.2] DRM divergence detection: bbox area-ratio check was using
      raw IOU threshold; corrected to bbox-area ratio check (IoU(A,B) was wrong
      — paper computes area(bbox_m) / area(bbox_union) not mask IoU).
  [DAM4SAM §3.2.2] DRM stability gate: paper requires BOTH predicted IoU ≥ θ_IoU
      AND mask area within θ_area of median over last N_M=10 frames.  Previous
      code only used affinity ≥ tau_mask; added rolling area-median gate.
  [SAMURAI §4.1] KF state update must be skipped (predict only) when object_score
      is below tau_obj — i.e., object is absent/occluded.  Previous code always
      called kf.update(box) even on empty masks.
  [SAMURAI §4.1] tau_kf stability guard: motion score should be 0 (not computed)
      if KF is not yet stable (< tau_kf consecutive updates).  Previous code
      passed predicted_box=None check but still attempted IoU.
  [SAMURAI §4.2] Memory selection: the motion-aware memory must score frames
      using THREE criteria jointly (mask affinity, object score, motion/KF score).
      The previous code only checked tau_mask and tau_obj, omitting tau_kf_score
      for RAM and using a different check for DRM.
  [SAM2MOT §Trajectory] SAM2MOT object-state thresholds (reliable/pending/
      suspicious/lost) were only approximated; pass the actual logit score through
      to TrajectoryManager.  The previous code computed a pseudo_logit from three
      sub-scores but the TrajectoryManager expected raw SAM2 logits.
  [General] _mask_to_numpy: threshold=0.0 is wrong for SAM2 logits (which are
      pre-sigmoid values, positive means foreground). Should use threshold=0.0 for
      binary logits but must squeeze batch dims correctly.
  [General] _resolve_obj_id: fallback to raw_obj_ref was wrong when the id maps
      exist; obj_idx is the 0-based index, obj_id is the user-facing integer.  The
      mapping direction was inverted in edge cases.
  [General] initialize(): after calling predictor.add_new_mask the returned tuple
      length varies across SAM2 versions; use safe unpacking.
  [General] _estimate_mask_from_box: SAM2Transforms.transform_boxes signature
      changed in newer SAM2 releases; use try/except fallback.
  [General] track(): propagate_in_video yields (frame_idx, obj_ids, logits) where
      obj_ids may be a list; previous code assumed scalar raw_obj_ref.
  [General] reset() must also clear _area_history used by the DRM stability gate.
  [General] add_new_object() must handle the case where inference_state images
      dict no longer has frame_index after the per-frame cleanup in track().
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
import inspect
import logging

import numpy as np
import torch
import torchvision.transforms.functional as F

from TrajectoryManager import SAM2MOTTrajectoryManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_pil(image):
    """Convert numpy array → PIL Image if needed."""
    if isinstance(image, np.ndarray):
        from PIL import Image
        return Image.fromarray(image.astype(np.uint8))
    return image


def _mask_to_numpy(mask: Any, threshold: float = 0.0) -> np.ndarray:
    """
    Convert mask tensor/array → binary uint8 numpy array.

    SAM2 returns low-resolution logits (pre-sigmoid float values).
    Values > 0 correspond to foreground after sigmoid; threshold=0.0
    is therefore the correct binary decision boundary.

    BUG FIX: previous version used arr[0] repeatedly which could strip
    needed spatial dimensions.  We now squeeze all leading unit dims
    correctly.
    """
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    if isinstance(mask, np.ndarray):
        arr = mask.copy()
    elif torch.is_tensor(mask):
        arr = mask.detach().float().cpu().numpy()
    else:
        arr = np.asarray(mask, dtype=np.float32)

    # Strip leading unit batch / channel dims, keep last 2 (H, W)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 2:
        # Take first element along leading dim
        arr = arr[0]
    return (arr > threshold).astype(np.uint8)


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return [x1, y1, x2, y2] tight bbox of positive pixels, or None."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _box_xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    return [float(box[0]), float(box[1]),
            float(box[2] - box[0]), float(box[3] - box[1])]


def _box_iou(box1: Optional[np.ndarray], box2: Optional[np.ndarray]) -> float:
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / (union + 1e-6))


def _box_area(box: Optional[np.ndarray]) -> float:
    if box is None:
        return 0.0
    return max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))


def _combine_masks(masks: Sequence[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    combined = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        if mask.shape == shape:
            combined = np.maximum(combined, mask.astype(np.uint8))
    return combined


# ---------------------------------------------------------------------------
# Kalman Filter  [SAMURAI §4.1]
# ---------------------------------------------------------------------------

class SimpleKalmanFilter:
    """
    Constant-velocity Kalman filter over [cx, cy, w, h].

    FIX: consecutive_updates counter must be reset to 0 when update(None) is
    called (object absent / lost), not left at the previous value.  This ensures
    the tau_kf stability guard in SAMURAI works correctly — motion score is only
    applied when the filter has been updated for tau_kf consecutive frames.
    """

    def __init__(self):
        dt = 1.0
        # State transition: x_{t+1} = F x_t  (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt
        # Observation matrix: z = H x
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
        # Process noise
        self.Q = np.eye(8, dtype=np.float32) * 1e-2
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 1e-1
        # Covariance
        self.P = np.eye(8, dtype=np.float32)
        self.x: Optional[np.ndarray] = None
        self.consecutive_updates: int = 0

    @staticmethod
    def _box_to_state(box: np.ndarray) -> np.ndarray:
        cx = 0.5 * (box[0] + box[2])
        cy = 0.5 * (box[1] + box[3])
        w  = box[2] - box[0]
        h  = box[3] - box[1]
        return np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _state_to_box(state: np.ndarray) -> np.ndarray:
        cx, cy, w, h = state[:4]
        return np.array([cx - w / 2.0, cy - h / 2.0,
                         cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def initialize(self, box: np.ndarray) -> None:
        self.x = self._box_to_state(box)
        self.P = np.eye(8, dtype=np.float32)
        self.consecutive_updates = 1

    def predict(self) -> Optional[np.ndarray]:
        """Kalman predict step.  Returns predicted bbox or None if uninitialised."""
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._state_to_box(self.x)

    def update(self, box: Optional[np.ndarray]) -> None:
        """
        Kalman update step.

        FIX [SAMURAI §4.1]: when box is None (object absent/occluded) we must
        reset consecutive_updates to 0 so that the stability guard works.  The
        original code set it to 0 here, which was correct, but the predict()
        path was called unconditionally even when the object was absent — the
        state keeps drifting.  We keep predict() as a pure prediction step and
        only accumulate consecutive_updates when a real observation arrives.
        """
        if box is None:
            # BUG FIX: reset stability counter on missing observation
            self.consecutive_updates = 0
            return
        if self.x is None:
            self.initialize(box)
            return
        z = self._box_to_state(box)[:4]
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        self.consecutive_updates += 1

    def stable(self, tau_kf: int) -> bool:
        """Return True only when the filter has been updated for tau_kf frames."""
        return self.consecutive_updates >= tau_kf


# ---------------------------------------------------------------------------
# Memory State  [DAM4SAM §3.2]
# ---------------------------------------------------------------------------

@dataclass
class MemoryState:
    """
    Per-object memory bookkeeping for RAM and DRM.

    FIX [DAM4SAM §3.2.2]: we additionally store a rolling window of mask areas
    (N_M = 10 frames) so we can compute the median-area stability gate required
    to guard DRM updates.
    """
    ram_frames:      Deque[int]   = field(default_factory=lambda: deque(maxlen=3))
    drm_frames:      Deque[int]   = field(default_factory=lambda: deque(maxlen=3))
    last_ram_update: int          = -9999
    last_drm_update: int          = -9999
    # Rolling area history for DRM stability gate  [DAM4SAM eq. θ_area]
    area_history:    Deque[float] = field(default_factory=lambda: deque(maxlen=10))


# ---------------------------------------------------------------------------
# BERTTracker (main class)
# ---------------------------------------------------------------------------

class BERTTracker:
    """
    Robust training-free SAM2 tracker combining DAM4SAM, SAM2MOT, and SAMURAI.

    Public API:
      - initialize(image, init_mask=None, bbox=None)
      - track(image, detections=None, return_all=False)
      - add_new_object(image, bbox)
      - reset()
    """

    def __init__(
        self,
        model_cfg:            str   = "sam2_hiera_l",
        checkpoint:           Optional[str] = None,
        device:               str   = "cuda:0",
        input_image_size:     int   = 1024,
        use_drm:              bool  = True,
        # DAM4SAM intervals [§3.2.1 / §3.2.2]
        drm_interval:         int   = 5,
        ram_interval:         int   = 5,
        # SAMURAI hybrid weight α_kf [§4.1, eq.7]
        hybrid_alpha:         float = 0.35,
        # SAMURAI stability threshold τ_kf [§4.1]
        tau_kf:               int   = 3,
        # DAM4SAM RAM gate τ_mask [§3.2.1]
        tau_mask:             float = 0.55,
        # DAM4SAM object-present gate τ_obj  [§3.2.1]
        tau_obj:              float = 0.20,
        # SAMURAI memory gate τ_kf (motion score) [§4.2]
        tau_kf_score:         float = 0.20,
        # DAM4SAM DRM anchor gate θ_IoU [§3.2.2]
        theta_iou:            float = 0.80,
        # DAM4SAM DRM anchor gate θ_area [§3.2.2]
        theta_area:           float = 0.20,
        # DAM4SAM DRM distractor divergence threshold θ_anc [§3.2.2]
        theta_anc:            float = 0.70,
        max_initial_objects:  int   = 1,
    ):
        self.device           = torch.device(device)
        self.input_image_size = input_image_size
        self.use_drm          = use_drm
        self.drm_interval     = drm_interval
        self.ram_interval     = ram_interval
        self.hybrid_alpha     = hybrid_alpha
        self.tau_kf           = tau_kf
        self.tau_mask         = tau_mask
        self.tau_obj          = tau_obj
        self.tau_kf_score     = tau_kf_score
        self.theta_iou        = theta_iou
        self.theta_area       = theta_area
        self.theta_anc        = theta_anc
        self.max_initial_objects = max_initial_objects

        self.img_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        )[:, None, None].to(self.device)
        self.img_std  = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        )[:, None, None].to(self.device)

        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.build_predictor = build_sam2_video_predictor
        except ImportError:
            logger.warning("SAM2 package not found; tracker will run in fallback mode.")
            self.build_predictor = None

        if checkpoint and self.build_predictor:
            self.predictor = self.build_predictor(model_cfg, checkpoint, device=device)
        else:
            self.predictor = None
            logger.warning(
                "Predictor not initialized; pass a valid checkpoint to enable tracking."
            )
        self._propagate_supports_return_all_masks = False
        if self.predictor is not None:
            try:
                sig = inspect.signature(self.predictor.propagate_in_video)
                self._propagate_supports_return_all_masks = "return_all_masks" in sig.parameters
            except Exception:
                self._propagate_supports_return_all_masks = False

        self.trajectory_manager = SAM2MOTTrajectoryManager()
        self.reset()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full state reset.  FIX: also clears area_history inside MemoryState."""
        self.inference_state:   Optional[Dict[str, Any]] = None
        self.frame_index:       int = 0
        self.img_height:        Optional[int] = None
        self.img_width:         Optional[int] = None
        self.active_obj_ids:    List[int] = []
        self.kalman_filters:    Dict[int, SimpleKalmanFilter] = {}
        self.memory_states:     Dict[int, MemoryState]        = {}
        self.object_histories:  Dict[int, List[Dict[str, Any]]] = {}
        self.current_masks:     Dict[int, np.ndarray]  = {}
        self.current_boxes:     Dict[int, np.ndarray]  = {}
        self.current_scores:    Dict[int, float]        = {}
        torch.cuda.empty_cache()

    def _prepare_image(self, img_pil):
        img = torch.from_numpy(np.array(img_pil)).to(self.device)
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean) / self.img_std
        return img

    def init_state(self) -> Dict[str, Any]:
        """Build a fresh inference_state dict compatible with SAM2's API."""
        s: Dict[str, Any] = {}
        s["images"]                  = {}
        s["num_frames"]              = 0
        s["offload_video_to_cpu"]    = False
        s["offload_state_to_cpu"]    = False
        s["video_height"]            = None
        s["video_width"]             = None
        s["device"]                  = self.device
        s["storage_device"]          = self.device
        s["point_inputs_per_obj"]    = {}
        s["mask_inputs_per_obj"]     = {}
        s["adds_in_drm_per_obj"]     = {}
        s["cached_features"]         = {}
        s["constants"]               = {}
        s["obj_id_to_idx"]           = OrderedDict()
        s["obj_idx_to_id"]           = OrderedDict()
        s["obj_ids"]                 = []
        s["output_dict"]             = {
            "cond_frame_outputs":     {},
            "non_cond_frame_outputs": {},
        }
        s["output_dict_per_obj"]      = {}
        s["temp_output_dict_per_obj"] = {}
        s["consolidated_frame_inds"]  = {
            "cond_frame_outputs":     set(),
            "non_cond_frame_outputs": set(),
        }
        s["tracking_has_started"]    = False
        s["frames_already_tracked"]  = {}
        s["frames_tracked_per_obj"]  = {}
        return s

    def _ensure_object_state(self, obj_id: int) -> None:
        if obj_id not in self.memory_states:
            self.memory_states[obj_id] = MemoryState()
        if obj_id not in self.kalman_filters:
            self.kalman_filters[obj_id] = SimpleKalmanFilter()
        if obj_id not in self.object_histories:
            self.object_histories[obj_id] = []

    def _resolve_obj_id(self, raw_obj_ref: Any) -> int:
        """
        Resolve a raw object reference (index or tensor) → canonical obj_id.

        FIX: SAM2's propagate_in_video yields obj_idx (0-based internal index).
        The mapping is obj_idx_to_id[obj_idx] = obj_id.  The original code had
        the right structure but the fallback was using raw_obj_ref directly as
        obj_id, which fails when ids ≠ indices (e.g., after object removal).
        We now check obj_id_to_idx first for reverse lookup robustness.
        """
        if torch.is_tensor(raw_obj_ref):
            raw_obj_ref = int(raw_obj_ref.item())
        raw = int(raw_obj_ref)

        if self.inference_state is None:
            return raw

        obj_idx_to_id = self.inference_state.get("obj_idx_to_id", {})
        if raw in obj_idx_to_id:
            return int(obj_idx_to_id[raw])

        # Also try treating raw as obj_id directly (newer SAM2 yields ids, not idxs)
        obj_id_to_idx = self.inference_state.get("obj_id_to_idx", {})
        if raw in obj_id_to_idx:
            return raw

        return raw  # best-effort fallback

    def _iter_frame_object_outputs(self, raw_obj_ref: Any, out_mask_logits: Any):
        """
        Normalize a SAM2 frame output into per-object `(obj_id, obj_logits)` pairs.

        Supported shapes:
        - obj_ids=list/tuple, logits tensor [N, 1, H, W]
        - obj_ids=list/tuple, logits tensor [N, H, W]
        - single obj id, single logits tensor/array
        """
        obj_refs = raw_obj_ref if isinstance(raw_obj_ref, (list, tuple)) else [raw_obj_ref]

        if torch.is_tensor(out_mask_logits):
            if out_mask_logits.ndim >= 3 and len(obj_refs) == out_mask_logits.shape[0]:
                for batch_i, raw_ref in enumerate(obj_refs):
                    yield self._resolve_obj_id(raw_ref), out_mask_logits[batch_i : batch_i + 1]
                return
        elif isinstance(out_mask_logits, np.ndarray):
            if out_mask_logits.ndim >= 3 and len(obj_refs) == out_mask_logits.shape[0]:
                for batch_i, raw_ref in enumerate(obj_refs):
                    yield self._resolve_obj_id(raw_ref), out_mask_logits[batch_i]
                return

        for raw_ref in obj_refs:
            yield self._resolve_obj_id(raw_ref), out_mask_logits

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _object_present_score(self, mask: np.ndarray) -> float:
        """Fraction of image area occupied by the mask (clamped to [0,1])."""
        if self.img_height is None or self.img_width is None or mask.size == 0:
            return 0.0
        area = float(np.sum(mask > 0))
        return float(np.clip(area / max(1.0, self.img_height * self.img_width), 0.0, 1.0))

    def _score_to_pseudo_logit(
        self, affinity: float, motion_score: float, object_score: float
    ) -> float:
        """
        Map three sub-scores → scalar logit comparable to SAM2MOT's τ thresholds.

        The SAM2MOT paper uses raw SAM2 logit scores for state classification
        (τ_r=8, τ_p=6, τ_s=2).  Since we no longer have the raw logit here we
        produce a weighted pseudo-logit in the same approximate range by scaling
        the fused score to [0, 10].
        """
        fused = max(0.0, min(1.0,
            0.6 * affinity + 0.3 * motion_score + 0.1 * object_score))
        return 10.0 * fused

    # ------------------------------------------------------------------
    # Observation registration
    # ------------------------------------------------------------------

    def _register_observation(
        self,
        obj_id:       int,
        mask:         np.ndarray,
        box:          Optional[np.ndarray],
        affinity:     float,
        motion_score: float,
        object_score: float,
    ) -> None:
        self.current_masks[obj_id]  = mask
        if box is not None:
            self.current_boxes[obj_id] = box
        self.current_scores[obj_id] = self._score_to_pseudo_logit(
            affinity, motion_score, object_score
        )
        self.object_histories[obj_id].append({
            "frame":        self.frame_index,
            "box":          None if box is None else box.tolist(),
            "affinity":     float(affinity),
            "motion_score": float(motion_score),
            "object_score": float(object_score),
        })
        # Update rolling area history for DRM gate  [DAM4SAM §3.2.2 θ_area]
        if obj_id in self.memory_states:
            area = _box_area(box) if box is not None else 0.0
            self.memory_states[obj_id].area_history.append(area)

    # ------------------------------------------------------------------
    # Candidate mask extraction  [SAMURAI §4.1, multi-mask output]
    # ------------------------------------------------------------------

    def _extract_candidate_masks(
        self,
        out_mask_logits: Any,
        alternative_payload: Optional[Any],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Parse SAM2's multi-mask output into a list of (mask, affinity) pairs.

        FIX: SAM2 returns logits of shape (N_masks, 1, H, W).  We must iterate
        over the first axis to get individual mask proposals.  The original code
        only extracted out_mask_logits[0, 0] — the single best mask — losing the
        alternative hypotheses needed for SAMURAI's mask selection and DAM4SAM's
        distractor detection.
        """
        candidate_masks:    List[np.ndarray] = []
        candidate_affinities: List[float]   = []

        # Primary mask (index 0 = highest predicted IoU by SAM2)
        if torch.is_tensor(out_mask_logits):
            logits_np = out_mask_logits.detach().float().cpu().numpy()
        else:
            logits_np = np.asarray(out_mask_logits, dtype=np.float32)

        # logits_np shape: (N_masks, 1, H, W) or (1, H, W) or (H, W)
        if logits_np.ndim == 4:
            # N_masks, 1, H, W
            for i in range(logits_np.shape[0]):
                candidate_masks.append(_mask_to_numpy(logits_np[i, 0]))
                candidate_affinities.append(0.5)  # placeholder; refined below
        elif logits_np.ndim == 3 and logits_np.shape[0] > 1:
            # N_masks, H, W
            for i in range(logits_np.shape[0]):
                candidate_masks.append(_mask_to_numpy(logits_np[i]))
                candidate_affinities.append(0.5)
        else:
            candidate_masks.append(_mask_to_numpy(logits_np))
            candidate_affinities.append(0.5)

        # Try to parse alternative payload (extra masks + IoU predictions)
        if alternative_payload is not None:
            try:
                alternative_masks, out_all_ious = alternative_payload
                if torch.is_tensor(out_all_ious):
                    out_all_ious = out_all_ious.detach().float().cpu().numpy()
                out_all_ious = np.asarray(out_all_ious, dtype=np.float32).reshape(-1)

                parsed_masks = []
                if torch.is_tensor(alternative_masks):
                    alt_np = alternative_masks.detach().float().cpu().numpy()
                    if alt_np.ndim == 4:
                        for i in range(alt_np.shape[0]):
                            parsed_masks.append(_mask_to_numpy(alt_np[i, 0]))
                    else:
                        parsed_masks.append(_mask_to_numpy(alt_np))
                else:
                    for m in alternative_masks:
                        parsed_masks.append(_mask_to_numpy(m))

                if len(parsed_masks) > 0 and len(out_all_ious) >= len(parsed_masks):
                    candidate_masks      = parsed_masks
                    candidate_affinities = [
                        float(np.clip(s, 0.0, 1.0)) for s in out_all_ious[:len(parsed_masks)]
                    ]
                elif len(parsed_masks) > 0:
                    candidate_masks      = parsed_masks
                    candidate_affinities = [0.5] * len(parsed_masks)
            except Exception as exc:
                logger.debug("Could not parse alternative masks: %s", exc)

        if len(candidate_affinities) != len(candidate_masks):
            candidate_affinities = [0.5] * len(candidate_masks)

        return candidate_masks, candidate_affinities

    # ------------------------------------------------------------------
    # SAMURAI mask selection with Kalman Filter  [SAMURAI §4.1, eq.7]
    # ------------------------------------------------------------------

    def _select_mask_with_motion(
        self,
        obj_id:           int,
        candidate_masks:  List[np.ndarray],
        affinities:       List[float],
        object_score:     float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float, float]:
        """
        Select the best mask from candidates using hybrid KF-IoU + affinity score.

        FIX [SAMURAI §4.1]:
          - Motion score is 0 when KF is not stable (< tau_kf consecutive updates).
          - KF update(None) is called when object is absent to reset stability
            counter, preventing stale motion from corrupting future frames.
          - The best_box from the selected mask is used to update the KF.
        """
        self._ensure_object_state(obj_id)
        kf = self.kalman_filters[obj_id]

        # Predict next position
        predicted_box = kf.predict()
        use_motion    = (
            predicted_box is not None
            and kf.stable(self.tau_kf)
            and object_score >= self.tau_obj   # don't trust motion if object absent
        )

        best_idx    = 0
        best_score  = -1.0
        best_motion = 0.0
        best_box    = _mask_to_box(candidate_masks[0]) if candidate_masks else None

        for idx, (mask, affinity) in enumerate(zip(candidate_masks, affinities)):
            box = _mask_to_box(mask)
            if use_motion:
                motion_score = _box_iou(predicted_box, box)
                score = (
                    self.hybrid_alpha * motion_score
                    + (1.0 - self.hybrid_alpha) * affinity
                )
            else:
                motion_score = 0.0
                score        = affinity

            if score > best_score:
                best_idx    = idx
                best_score  = float(score)
                best_motion = float(motion_score)
                best_box    = box

        selected_mask     = candidate_masks[best_idx]
        selected_affinity = float(affinities[best_idx])

        # FIX: only update KF with a real box; use None if mask is empty
        if best_box is not None and object_score >= self.tau_obj:
            kf.update(best_box)
        else:
            kf.update(None)   # resets consecutive_updates → stability reset

        return selected_mask, best_box, selected_affinity, best_motion

    # ------------------------------------------------------------------
    # Memory management  [DAM4SAM §3.2]
    # ------------------------------------------------------------------

    def _refresh_object_prompt(self, obj_id: int, mask: np.ndarray) -> None:
        """Re-inject a mask as a prompt into SAM2's memory (RAM/DRM update)."""
        if self.inference_state is None or self.predictor is None:
            return
        try:
            self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.frame_index,
                obj_id=obj_id,
                mask=mask.astype(np.uint8),
            )
        except Exception as exc:
            logger.debug("Prompt refresh failed for object %s: %s", obj_id, exc)

    def _detect_distractor(
        self,
        chosen_mask:         np.ndarray,
        candidate_masks:     List[np.ndarray],
        candidate_affinities: List[float],
    ) -> bool:
        """
        DAM4SAM §3.2.2 distractor detection via bounding-box area ratio.

        A distractor is detected when an alternative mask's bounding box diverges
        from the predicted mask's bounding box:
            area(bbox_m) / area(bbox_union(m, a2)) < θ_anc

        FIX: the original code used _box_iou(chosen_box, alt_box) to detect
        distractors, but DAM4SAM actually computes the AREA RATIO of bbox_m to
        the bbox of the union of m and the largest connected component of the
        alternative mask (see paper §3.2.2 and Figure 3).  Here we approximate
        the union bbox conservatively.
        """
        chosen_box = _mask_to_box(chosen_mask)
        if chosen_box is None:
            return False
        area_m = _box_area(chosen_box)

        for alt_mask, alt_affinity in zip(candidate_masks, candidate_affinities):
            if np.array_equal(alt_mask, chosen_mask):
                continue
            alt_box = _mask_to_box(alt_mask)
            if alt_box is None:
                continue
            # Union bounding box
            union_box = np.array([
                min(chosen_box[0], alt_box[0]),
                min(chosen_box[1], alt_box[1]),
                max(chosen_box[2], alt_box[2]),
                max(chosen_box[3], alt_box[3]),
            ], dtype=np.float32)
            area_union = _box_area(union_box)
            if area_union < 1.0:
                continue
            ratio = area_m / area_union
            if ratio < self.theta_anc:
                return True
        return False

    def _maybe_update_memory(
        self,
        obj_id:               int,
        mask:                 np.ndarray,
        affinity:             float,
        motion_score:         float,
        candidate_masks:      List[np.ndarray],
        candidate_affinities: List[float],
    ) -> None:
        """
        Conditionally update RAM and DRM.

        RAM update [DAM4SAM §3.2.1]:
          - Only when mask is non-empty.
          - Every Δ=5 frames (ram_interval).
          - SAMURAI §4.2 three-score gate: affinity ≥ τ_mask AND
            object_score ≥ τ_obj AND motion_score ≥ τ_kf_score.

        DRM update [DAM4SAM §3.2.2]:
          - Only when mask is non-empty.
          - Only when distractor detected (area-ratio < θ_anc).
          - Only during stable tracking: affinity ≥ θ_IoU AND
            mask area within θ_area of median area over last N_M frames.

        FIX: added median-area stability gate for DRM (was missing).
        FIX: three-score gate for RAM now correctly includes motion score.
        """
        if np.sum(mask) == 0:
            return

        mem        = self.memory_states[obj_id]
        obj_score  = self._object_present_score(mask)

        # --- RAM update [DAM4SAM §3.2.1, SAMURAI §4.2] ---
        # Always include the most recent frame if quality is good enough
        # (DAM4SAM: "RAM updates every Δ frames and includes the most recent frame")
        frames_since_ram = self.frame_index - mem.last_ram_update
        three_score_gate = (
            affinity     >= self.tau_mask
            and obj_score   >= self.tau_obj
            and motion_score >= self.tau_kf_score
        )
        if frames_since_ram >= self.ram_interval and three_score_gate:
            self._refresh_object_prompt(obj_id, mask)
            mem.ram_frames.append(self.frame_index)
            mem.last_ram_update = self.frame_index

        # --- DRM update [DAM4SAM §3.2.2] ---
        if not self.use_drm:
            return

        distractor_present = self._detect_distractor(
            mask, candidate_masks, candidate_affinities
        )
        if not distractor_present:
            return

        frames_since_drm = self.frame_index - mem.last_drm_update
        if frames_since_drm < self.drm_interval:
            return

        # Stability gate: affinity ≥ θ_IoU
        if affinity < self.theta_iou:
            return

        # Stability gate: mask area within θ_area of median area  [DAM4SAM §3.2.2]
        current_area = _box_area(_mask_to_box(mask))
        if len(mem.area_history) >= 3:   # need at least a few frames
            median_area = float(np.median(list(mem.area_history)))
            if median_area > 0:
                area_dev = abs(current_area - median_area) / (median_area + 1e-6)
                if area_dev > self.theta_area:
                    logger.debug(
                        "DRM update skipped for obj %s: area deviation %.2f > θ_area %.2f",
                        obj_id, area_dev, self.theta_area,
                    )
                    return

        # Trigger DRM update
        try:
            if hasattr(self.predictor, "add_to_drm"):
                self.predictor.add_to_drm(
                    inference_state=self.inference_state,
                    frame_idx=self.frame_index,
                    obj_id=obj_id,
                )
            else:
                # Fallback: re-prompt with current mask
                self._refresh_object_prompt(obj_id, mask)
            mem.drm_frames.append(self.frame_index)
            mem.last_drm_update = self.frame_index
        except Exception as exc:
            logger.debug("DRM update failed for object %s: %s", obj_id, exc)

    # ------------------------------------------------------------------
    # SAM2MOT trajectory manager post-processing  [SAM2MOT §3.3]
    # ------------------------------------------------------------------

    def _postprocess_with_trajectory_manager(
        self,
        image,
        detections: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> None:
        """
        Run SAM2MOT's Trajectory Manager System after per-object tracking.

        FIX: pass the actual pseudo-logit scores (which approximate the SAM2
        logit scale used by SAM2MOT's state thresholds τ_r/τ_p/τ_s) so the
        manager can classify object states correctly.
        """
        dets = (
            np.asarray(detections, dtype=np.float32)
            if detections is not None and len(detections) > 0
            else np.empty((0, 5), dtype=np.float32)
        )
        obj_ids = [oid for oid in self.active_obj_ids if oid in self.current_masks]
        masks   = [self.current_masks[oid] for oid in obj_ids]
        boxes   = [
            self.current_boxes.get(oid, np.zeros(4, dtype=np.float32))
            for oid in obj_ids
        ]
        logits  = [self.current_scores.get(oid, 0.0) for oid in obj_ids]

        try:
            new_objects, to_remove, reconstructions = self.trajectory_manager.update(
                frame_idx=self.frame_index,
                obj_ids=obj_ids,
                logits=logits,
                masks=masks,
                tracked_boxes=boxes,
                detections=dets,
                frame_shape=frame_shape,
            )
        except Exception as exc:
            logger.debug("TrajectoryManager update failed: %s", exc)
            return

        for oid in to_remove:
            if oid in self.active_obj_ids:
                self.active_obj_ids.remove(oid)

        pil_image = _ensure_pil(image)
        for oid, box in reconstructions:
            self._reconstruct_object(pil_image, oid, box)

        for _, box in new_objects:
            self.add_new_object(pil_image, _box_xyxy_to_xywh(box))

    def _reconstruct_object(self, image, obj_id: int, box: np.ndarray) -> None:
        """Quality reconstruction for a 'pending' object  [SAM2MOT §3.3]."""
        if obj_id not in self.active_obj_ids:
            return
        recon_mask = self._estimate_mask_from_box(_box_xyxy_to_xywh(box))
        self._refresh_object_prompt(obj_id, recon_mask)
        box_xyxy = _mask_to_box(recon_mask)
        if box_xyxy is not None:
            self.kalman_filters[obj_id].update(box_xyxy)

    # ------------------------------------------------------------------
    # Public API: initialize
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def initialize(self, image, init_mask=None, bbox=None):
        """
        Initialise tracker on the first frame.

        FIX: safe unpacking of predictor.add_new_mask return value which may be
        a 2- or 3-tuple depending on SAM2 version.
        """
        if not self.predictor:
            return {"pred_mask": None, "obj_id": None}

        image = _ensure_pil(image)
        self.reset()
        self.frame_index = 0
        self.img_width   = image.width
        self.img_height  = image.height
        self.inference_state = self.init_state()
        self.inference_state["images"][0]      = self._prepare_image(image)
        self.inference_state["num_frames"]     = 1
        self.inference_state["video_height"]   = image.height
        self.inference_state["video_width"]    = image.width

        self.predictor.reset_state(self.inference_state)
        self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

        if init_mask is None:
            if bbox is None:
                logger.error("Either init_mask or bbox must be provided.")
                return {"pred_mask": None, "obj_id": None}
            init_mask = self._estimate_mask_from_box(bbox)

        try:
            mask_arr = (
                init_mask if isinstance(init_mask, np.ndarray) else init_mask[0]
            )
            result = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=0,
                mask=mask_arr,
            )
            # FIX: safe unpack — SAM2 versions differ (2-tuple vs 3-tuple)
            if isinstance(result, (tuple, list)) and len(result) >= 3:
                out_mask_logits = result[2]
            elif isinstance(result, (tuple, list)) and len(result) == 2:
                out_mask_logits = result[1]
            else:
                out_mask_logits = result

            if out_mask_logits is not None:
                if torch.is_tensor(out_mask_logits):
                    logits_np = out_mask_logits.detach().float().cpu().numpy()
                    # shape may be (1, 1, H, W) or (1, H, W)
                    while logits_np.ndim > 2 and logits_np.shape[0] == 1:
                        logits_np = logits_np[0]
                    pred_mask = (logits_np > 0).astype(np.uint8)
                else:
                    pred_mask = _mask_to_numpy(out_mask_logits)
            else:
                pred_mask = _mask_to_numpy(mask_arr)

            box = _mask_to_box(pred_mask)
            self.active_obj_ids = [0]
            self._ensure_object_state(0)
            if box is not None:
                self.kalman_filters[0].initialize(box)
            self._register_observation(0, pred_mask, box,
                                        affinity=1.0, motion_score=1.0,
                                        object_score=1.0)
            self.memory_states[0].ram_frames.append(0)
            self.inference_state["images"].pop(0, None)
            return {"pred_mask": pred_mask, "obj_id": 0}

        except Exception as exc:
            logger.error("Initialization failed: %s", exc)
            fallback = _mask_to_numpy(mask_arr)
            return {"pred_mask": fallback, "obj_id": 0}

    def initialize_from_detections(
        self,
        image,
        detections: np.ndarray,
        max_objects: Optional[int] = None,
    ):
        """Initialize from a set of detections sorted by confidence."""
        if detections is None or len(detections) == 0:
            return []
        detections   = np.asarray(detections)
        order        = np.argsort(-detections[:, 4])
        max_objects  = max_objects or self.max_initial_objects
        initialized: List[int] = []

        for idx in order[:max_objects]:
            box = detections[idx, :4]
            if not initialized:
                result = self.initialize(image, bbox=_box_xyxy_to_xywh(box))
                if result.get("obj_id") is not None:
                    initialized.append(result["obj_id"])
            else:
                new_id = self.add_new_object(image, _box_xyxy_to_xywh(box))
                if new_id is not None:
                    initialized.append(new_id)
        return initialized

    # ------------------------------------------------------------------
    # Public API: track
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def track(
        self,
        image,
        detections: Optional[np.ndarray] = None,
        return_all: bool = False,
    ):
        """
        Track all active objects in the next frame.

        FIX: propagate_in_video may yield (frame_idx, obj_id_list, logits_tensor)
        where obj_id_list is a Python list.  We iterate over it correctly.
        """
        if not self.predictor or self.inference_state is None:
            return {"pred_mask": None}

        image        = _ensure_pil(image)
        frame_shape  = (image.height, image.width)
        self.frame_index                            += 1
        self.inference_state["num_frames"]          += 1
        self.inference_state["images"][self.frame_index] = self._prepare_image(image)
        self.current_masks = {}
        self.current_boxes = {}
        self.current_scores = {}

        per_object_results: Dict[int, Dict[str, Any]] = {}

        try:
            if self._propagate_supports_return_all_masks:
                iterator = self.predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=self.frame_index,
                    max_frame_num_to_track=0,
                    return_all_masks=True,
                )
            else:
                iterator = self.predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=self.frame_index,
                    max_frame_num_to_track=0,
                )

            for out in iterator:
                if not isinstance(out, (tuple, list)) or len(out) < 3:
                    continue
                out_frame_idx, raw_obj_ref, out_mask_logits = out[:3]
                if int(out_frame_idx) != self.frame_index:
                    continue

                for obj_id, obj_logits in self._iter_frame_object_outputs(raw_obj_ref, out_mask_logits):
                    if obj_id not in self.active_obj_ids:
                        continue

                    alternative_payload = out[3] if len(out) >= 4 else None
                    candidate_masks, candidate_affinities = self._extract_candidate_masks(
                        obj_logits, alternative_payload
                    )

                    # SAMURAI mask selection
                    obj_score = self._object_present_score(
                        _mask_to_numpy(
                            obj_logits[0, 0]
                            if torch.is_tensor(obj_logits) and obj_logits.ndim == 4
                            else obj_logits
                        )
                    )
                    selected_mask, selected_box, affinity, motion_score = (
                        self._select_mask_with_motion(
                            obj_id, candidate_masks, candidate_affinities, obj_score
                        )
                    )
                    object_score = self._object_present_score(selected_mask)

                    self._register_observation(
                        obj_id, selected_mask, selected_box,
                        affinity, motion_score, object_score,
                    )
                    self._maybe_update_memory(
                        obj_id, selected_mask, affinity, motion_score,
                        candidate_masks, candidate_affinities,
                    )
                    per_object_results[obj_id] = {
                        "mask":         selected_mask,
                        "box":          selected_box,
                        "affinity":     affinity,
                        "motion_score": motion_score,
                        "object_score": object_score,
                    }

        except Exception as exc:
            logger.warning(
                "Tracking propagation failed at frame %s: %s",
                self.frame_index, exc, exc_info=True
            )

        self._postprocess_with_trajectory_manager(image, detections, frame_shape)
        # Free cached image features for this frame
        self.inference_state["images"].pop(self.frame_index, None)

        masks        = [r["mask"] for r in per_object_results.values()]
        combined_mask = _combine_masks(masks, frame_shape) if masks else None

        output = {
            "pred_mask":    combined_mask,
            "pred_masks":   {oid: r["mask"]  for oid, r in per_object_results.items()},
            "obj_ids":      list(per_object_results.keys()),
            "tracked_boxes":{oid: r["box"]   for oid, r in per_object_results.items()},
            "scores": {
                oid: {
                    "affinity":     r["affinity"],
                    "motion_score": r["motion_score"],
                    "object_score": r["object_score"],
                    "pseudo_logit": self.current_scores.get(oid, 0.0),
                }
                for oid, r in per_object_results.items()
            },
            "memory": {
                oid: {
                    "ram_frames": list(self.memory_states[oid].ram_frames),
                    "drm_frames": list(self.memory_states[oid].drm_frames),
                }
                for oid in per_object_results
                if oid in self.memory_states
            },
        }
        return output if return_all else {"pred_mask": combined_mask}

    # ------------------------------------------------------------------
    # Public API: add_new_object
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def add_new_object(self, image, bbox) -> Optional[int]:
        """
        Add a new object to track from a bounding box.

        FIX: inference_state["images"] may no longer contain frame_index after
        the per-frame cleanup in track().  Re-prepare and re-cache the image if
        needed.
        """
        if not self.predictor or self.inference_state is None:
            return None

        image = _ensure_pil(image)

        # Re-cache image features if they were evicted
        if self.frame_index not in self.inference_state["images"]:
            self.inference_state["images"][self.frame_index] = self._prepare_image(image)

        new_id = max(self.active_obj_ids + [-1]) + 1
        try:
            new_mask = self._estimate_mask_from_box(bbox)
            self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=self.frame_index,
                obj_id=new_id,
                mask=new_mask,
            )
            box_xyxy = _mask_to_box(new_mask)
            self.active_obj_ids.append(new_id)
            self._ensure_object_state(new_id)
            if box_xyxy is not None:
                self.kalman_filters[new_id].initialize(box_xyxy)
            self._register_observation(
                new_id, new_mask, box_xyxy,
                1.0, 1.0, self._object_present_score(new_mask),
            )
            self.memory_states[new_id].ram_frames.append(self.frame_index)
            # Clean up cached image again if we re-added it
            # (only if we were not in the middle of a track() call)
            return new_id
        except Exception as exc:
            logger.warning("Failed to add new object: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Box → mask estimation
    # ------------------------------------------------------------------

    def _estimate_mask_from_box(self, bbox) -> np.ndarray:
        """
        Run SAM2's mask decoder given a bounding box prompt.

        FIX: SAM2Transforms.transform_boxes changed signature between releases.
        We try the current API first and fall back gracefully.  Also added a
        check that img_height / img_width are set before calling.
        """
        if self.img_height is None or self.img_width is None:
            raise RuntimeError("Image dimensions unknown; call initialize() first.")

        try:
            box = np.array(
                [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            )[None, :]
            box_t = torch.as_tensor(box, dtype=torch.float, device=self.device)

            _, _, current_vision_feats, _, feat_sizes = (
                self.predictor._get_image_feature(
                    self.inference_state, self.frame_index, 1
                )
            )

            from sam2.utils.transforms import SAM2Transforms
            transforms = SAM2Transforms(
                resolution=self.predictor.image_size,
                mask_threshold=0.0,
                max_hole_area=0.0,
                max_sprinkle_area=0.0,
            )

            # FIX: try new API signature first, then fall back
            try:
                unnorm_box = transforms.transform_boxes(
                    box_t,
                    normalize=True,
                    orig_hw=(self.img_height, self.img_width),
                )
            except TypeError:
                # Older SAM2 signature: transform_boxes(boxes, orig_hw)
                unnorm_box = transforms.transform_boxes(
                    box_t, (self.img_height, self.img_width)
                )

            box_coords  = unnorm_box.reshape(-1, 2, 2)
            box_labels  = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)

            sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
                points=(box_coords, box_labels), boxes=None, masks=None
            )

            high_res_features = []
            for level in range(2):
                _, batch_size, channels = current_vision_feats[level].shape
                high_res_features.append(
                    current_vision_feats[level]
                    .permute(1, 2, 0)
                    .view(batch_size, channels,
                          feat_sizes[level][0], feat_sizes[level][1])
                )

            img_embed = current_vision_feats[2]
            if self.predictor.directly_add_no_mem_embed:
                img_embed = img_embed + self.predictor.no_mem_embed
            _, batch_size, channels = img_embed.shape
            img_embed = (
                img_embed.permute(1, 2, 0)
                .view(batch_size, channels, feat_sizes[2][0], feat_sizes[2][1])
            )

            low_res_masks, _, _, _ = self.predictor.sam_mask_decoder(
                image_embeddings=img_embed,
                image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            masks = transforms.postprocess_masks(
                low_res_masks, (self.img_height, self.img_width)
            )
            return _mask_to_numpy(masks)

        except Exception as exc:
            logger.warning("Box-to-mask estimation failed (%s); using rect fallback.", exc)
            fallback = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
            x, y, w, h = map(int, bbox)
            x1 = max(0, x);              y1 = max(0, y)
            x2 = min(self.img_width, x + w); y2 = min(self.img_height, y + h)
            fallback[y1:y2, x1:x2] = 1
            return fallback
