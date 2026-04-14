import os
import pandas as pd
from ultralytics import SAM

# 1. Configuration
dataset_root = "/home/cs24m118/datasets/MOT17/train" 
model_path = "sam2.1_b.pt"
output_dir = "sam_mot17_results"
# Directory where annotated videos will be stored
video_output_dir = "sam_mot17_videos" 

os.makedirs(output_dir, exist_ok=True)

# Load SAM 2.1 model
model = SAM(model_path)

# 2. Iterate through all sequences
sequences = [s for s in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, s))]

for seq in sequences:
    print(f"--- Processing sequence: {seq} ---")
    
    seq_path = os.path.join(dataset_root, seq)
    det_file = os.path.join(seq_path, "det/det.txt")
    img_folder = os.path.join(seq_path, "img1")
    
    if not os.path.exists(det_file):
        continue

    detections = pd.read_csv(det_file, header=None)
    res_file = open(os.path.join(output_dir, f"{seq}.txt"), "w")
    images = sorted(os.listdir(img_folder))

    # To save as a video, we collect frames or let Ultralytics handle saving per frame.
    # Note: Running predict on a folder is more efficient for video saving.
    for i, img_name in enumerate(images):
        frame_id = i + 1
        img_path = os.path.join(img_folder, img_name)
        frame_dets = detections[detections[0] == frame_id]
        
        if frame_dets.empty:
            continue

        bboxes = []
        for _, row in frame_dets.iterrows():
            x, y, w, h = row[2], row[3], row[4], row[5]
            bboxes.append([x, y, x + w, y + h])

        # 3. Run SAM inference + SAVE IMAGE/FRAME
        # save=True will save the annotated frames to video_output_dir/{seq}/
        results = model.predict(
            source=img_path, 
            bboxes=bboxes, 
            verbose=False, 
            retina_masks=True,
            save=True, 
            project=video_output_dir,
            name=seq,
            exist_ok=True # Keeps frames in the same sequence folder
        )

        # 4. Save results to MOT .txt format
        if results and results[0].boxes:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                x, y, x2, y2 = xyxy
                w, h = x2 - x, y2 - y
                conf = box.conf[0].item()
                res_file.write(f"{frame_id},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")

    res_file.close()
    
    # Optional: Convert saved frames to an actual .mp4 using OpenCV or FFmpeg here 
    # if you want a single file instead of a folder of images.
    print(f"Finished {seq}. Results and annotated frames saved.")

print("\nAll sequences processed.")