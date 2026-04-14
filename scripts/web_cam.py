import cv2
cap = cv2.VideoCapture(0)
hybrid = InteractiveHybridSAM2()          # tiny/base model for 30 fps
hybrid.init_video(None)                   # live mode

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    # Optional lightweight detector (every 5 frames)
    dets = your_detector(frame) if frame_idx % 5 == 0 else None

    # Propagate hybrid tracker
    masks, _, _, boxes = hybrid.propagate_frame(frame_idx, frame.shape[:2], detections=dets)

    # Draw
    vis = frame.copy()
    for oid, mask in masks.items():
        color = (0, 255, 0)
        vis[mask > 0] = vis[mask > 0] * 0.5 + np.array(color) * 0.5
        cv2.putText(vis, f"ID{oid}", (int(boxes[oid][0]), int(boxes[oid][1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Hybrid SAM2 + SAMURAI + DAM4SAM + SAM2MOT", vis)

    # User click = new object
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Mouse click handling → call hybrid.add_user_prompt(frame_idx, points=click_point)

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()