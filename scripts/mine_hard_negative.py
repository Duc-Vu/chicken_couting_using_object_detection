import os
import json
import cv2
import joblib

from src.features.hog import extract_hog
from src.training.hard_negative import mine_hard_negatives


TRAIN_IMG_DIR = "data/raw/train/images"
TRAIN_COCO = "data/raw/train/annotations/coco.json"
OUT_DIR = "data/processed/hard_negative"
MODEL_PATH = "models/svm_chicken.pkl"

WINDOW_SIZE = 64
STEP = 16
SCORE_THRESH = 0.0
MAX_HNEG_PER_IMAGE = 10


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    svm = joblib.load(MODEL_PATH)

    with open(TRAIN_COCO, "r", encoding="utf-8") as f:
        coco = json.load(f)

    gt_map = {}
    for ann in coco["annotations"]:
        gt_map.setdefault(ann["image_id"], []).append(ann["bbox"])

    hneg_id = 35271

    for img_info in coco["images"]:
        img_path = os.path.join(
            TRAIN_IMG_DIR, img_info["file_name"]
        )
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_bboxes = gt_map.get(img_info["id"], [])

        hneg_id = mine_hard_negatives(
            img=img,
            svm=svm,
            feature_extractor=extract_hog,
            gt_bboxes=gt_bboxes,
            output_dir=OUT_DIR,
            start_id=hneg_id,
            window_size=WINDOW_SIZE,
            step=STEP,
            score_thresh=SCORE_THRESH,
            max_per_image=MAX_HNEG_PER_IMAGE,
        )

    print("DONE. Total hard negatives:", hneg_id)
