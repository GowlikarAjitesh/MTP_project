import json
import os
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

def rle_to_bbox(rle, height, width):
    """
    Convert COCO RLE to bounding box (x, y, w, h)
    """
    mask = mask_utils.decode(rle)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def convert_sav_json_to_mot(
    sav_json_path,
    output_gt_path
):
    with open(sav_json_path, "r") as f:
        data = json.load(f)

    video_h = int(data["video_height"])
    video_w = int(data["video_width"])

    masklets = data["masklet"]
    masklet_ids = data["masklet_id"]
    first_frames = data["masklet_first_appeared_frame"]

    mot_lines = []

    for obj_idx, (track_id, start_frame) in enumerate(
        zip(masklet_ids, first_frames)
    ):
        rles = masklets[obj_idx]
        start_frame = int(start_frame)

        for i, rle in enumerate(rles):
            frame_id = start_frame + i + 1  # MOT is 1-based

            bbox = rle_to_bbox(
                rle,
                height=video_h,
                width=video_w
            )
            if bbox is None:
                continue

            x, y, w, h = bbox

            # MOT format:
            # frame, id, x, y, w, h, conf, class, visibility
            mot_lines.append(
                f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1"
            )

    os.makedirs(os.path.dirname(output_gt_path), exist_ok=True)

    with open(output_gt_path, "w") as f:
        f.write("\n".join(mot_lines))

    print(f"✅ Saved MOT GT: {output_gt_path}")
    print(f"Total GT boxes: {len(mot_lines)}")


if __name__ == "__main__":
    sav_json = "/home/cs24m118/sav_dataset/sav_train/sav_000/sav_000001_manual.json"
    output_gt = "/home/cs24m118/sav_dataset/gt_mot/sav_000001/gt.txt"

    convert_sav_json_to_mot(sav_json, output_gt)
