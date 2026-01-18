import os
import cv2
import json
import joblib
from ..features.hog import extract_hog

# ================= CONFIG =================
TRAIN_IMG_DIR = "data/raw/train/images"
TRAIN_COCO = "data/raw/train/annotations/coco.json"

OUT_DIR = "data/processed/hard_negative"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "models/svm_chicken.pkl"

WINDOW_SIZE = 64
STEP = 16
SCORE_THRESH = 0.0 
MAX_HNEG_PER_IMAGE = 10


# ================= IOU CHECK ==============
def overlap_with_any_gt(x, y, w, h, gt_bboxes):
    
    for bx, by, bw, bh in gt_bboxes:
        if not (x + w < bx or x > bx + bw or y + h < by or y > by + bh):
            return True
    return False

# ================= LOAD ====================
svm = joblib.load(MODEL_PATH)

with open(TRAIN_COCO, "r", encoding="utf-8") as f:
    coco = json.load(f)


gt_map = {}
for ann in coco["annotations"]:
    gt_map.setdefault(ann["image_id"], []).append(ann["bbox"])

# ================= MAIN ====================
hneg_id = 0

for img_info in coco["images"]:
    img_path = os.path.join(TRAIN_IMG_DIR, img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W = img.shape[:2]
    gt_bboxes = gt_map.get(img_info["id"], [])

    count_this_img = 0

    for y in range(0, H - WINDOW_SIZE, STEP):
        for x in range(0, W - WINDOW_SIZE, STEP):
            patch = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]

            feat = extract_hog(patch).reshape(1, -1)
            score = svm.decision_function(feat)[0]

            if score > SCORE_THRESH:
               
                if not overlap_with_any_gt(x, y, WINDOW_SIZE, WINDOW_SIZE, gt_bboxes):
                    out_path = os.path.join(
                        OUT_DIR, f"hneg_{hneg_id}.png"
                    )
                    cv2.imwrite(out_path, patch)
                    print(f"HardNeg {hneg_id} | score={score:.2f}")

                    hneg_id += 1
                    count_this_img += 1

                    if count_this_img >= MAX_HNEG_PER_IMAGE:
                        break
        if count_this_img >= MAX_HNEG_PER_IMAGE:
            break

print("DONE. Total hard negatives:", hneg_id)
