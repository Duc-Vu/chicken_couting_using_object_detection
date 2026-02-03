import cv2
import joblib

from src.svm_detector.features.hog import extract_hog
from src.svm_detector.inference.nms import nms
import numpy as np
from config import config

# ========== CONFIG ==========     
BASE_WINDOW = config.WIN_SIZE  
STEP = 16
SCORE_THRESH = 0.7     
IOU_THRESH = 0.3

SCALE = 0.5              
VIDEO_PATH = "dataset/test_video/test_chicken.mp4"
MODEL_PATH = "models/svm/svm_chicken_v1.4_base.pkl"
SAVE_NEGATIVE_PATH = "dataset/svm/train/hard_negative"

SEEK_STEP = 30
FPS_SCALE = 1.0
# ========== LOAD ==========
svm = joblib.load(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

paused = False


def detect_chickens(frame):
    H, W = frame.shape[:2]
    boxes, scores = [], []

    win_w, win_h = BASE_WINDOW

    for y in range(0, H - win_h, STEP):
        for x in range(0, W - win_w, STEP):

            patch = frame[y:y + win_h, x:x + win_w]

            if patch.shape[0] != win_h or patch.shape[1] != win_w:
                continue

            feat = extract_hog(patch).reshape(1, -1)
            score = svm.decision_function(feat)[0]

            if score > SCORE_THRESH:
                boxes.append([x, y, x + win_w, y + win_h])
                scores.append(score)

    if len(boxes) == 0:
        return []

    keep = nms(boxes, scores, iou_thresh=IOU_THRESH)
    return [boxes[i] for i in keep]


# ========== MAIN LOOP ==========

hneg_id = 0
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- resize video theo tỉ lệ ----
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        final_boxes = detect_chickens(frame)

        # ---- draw boxes ----
        for (x1, y1, x2, y2) in final_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Chicken count: {len(final_boxes)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Scale: {SCALE} | Thresh: {SCORE_THRESH}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("Chicken Counting", frame)

    # ===== Keyboard control =====
    key = cv2.waitKey(int(30 / FPS_SCALE)) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        paused = not paused
        state = "PAUSED" if paused else "UNPAUSED"
        print(state)

    elif key == ord('d'):
        cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur + SEEK_STEP)
        print(f"SKIP {SEEK_STEP}")

    elif key == ord('a'):
        cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - SEEK_STEP))
        print(f"BACK {SEEK_STEP}")

    elif key == ord('w'):
        FPS_SCALE = min(4.0, FPS_SCALE * 2)

    elif key == ord('s'):
        FPS_SCALE = max(0.25, FPS_SCALE / 2)
        
    elif key == ord('h'):
        last_boxes = final_boxes
        last_frame = frame.copy()
        for (x1, y1, x2, y2) in last_boxes:
            patch = last_frame[y1:y2, x1:x2]
            patch = cv2.resize(patch, BASE_WINDOW)

            cv2.imwrite(
                f"{SAVE_NEGATIVE_PATH}/hneg_{hneg_id}.png",
                patch
            )
            hneg_id += 1

        print("Saved hard negatives")


cap.release()
cv2.destroyAllWindows()
