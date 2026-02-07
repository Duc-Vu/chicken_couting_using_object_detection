import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from tqdm import tqdm

from src.svm_detector.features.hog import extract_hog
from src.svm_detector.inference.sliding_window import run_svm_inference
from src.svm_detector.inference.nms import nms
from config import config

# ---------------- CONFIG ----------------
WINDOW_SIZE = config.WIN_SIZE
STEP = 16
THRESHOLD = 0.5
IS_NMS = True

IMAGE_DIR = "dataset/yolo/test/images"
LABEL_DIR = "dataset/yolo/test/labels"

SVM_PATH = "models/svm/svm_chicken_v1.5_base.pkl"


# ---------------- UTILS ----------------
def apply_nms(detections, score_thresh=0.0, nms_thresh=0.4):
    if len(detections) == 0:
        return []

    boxes = []
    scores = []

    for det in detections:
        x1, y1, x2, y2 = det[:4]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(det[4]))

    indices = nms(boxes, scores, score_thresh, nms_thresh)
    if len(indices) == 0:
        return []

    return [detections[i] for i in indices.flatten()]


def count_gt_from_yolo(label_path):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r") as f:
        return len(f.readlines())


# ---------------- MAIN ----------------
if __name__ == "__main__":
    svm = joblib.load(SVM_PATH)

    gt_counts = []
    pred_counts = []

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ]

    for img_name in tqdm(
        image_files,
        desc="Evaluating counting",
        unit="image"
    ):
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        if img is None:
            continue

        # --- inference ---
        detections = run_svm_inference(
            img=img,
            svm=svm,
            feature_extractor=extract_hog,
            window_size=WINDOW_SIZE,
            step=STEP,
            threshold=THRESHOLD,
        )

        if IS_NMS:
            detections = apply_nms(
                detections,
                score_thresh=THRESHOLD,
                nms_thresh=0.1,
            )

        # --- counting ---
        pred_count = len(detections)
        gt_count = count_gt_from_yolo(label_path)

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

    # ---------------- METRICS ----------------
    mae = mean_absolute_error(gt_counts, pred_counts)
    rmse = np.sqrt(((np.array(gt_counts) - np.array(pred_counts)) ** 2).mean())

    print("===================================")
    print(f"Test images     : {len(gt_counts)}")
    print(f"Counting MAE    : {mae:.4f}")
    print(f"Counting RMSE   : {rmse:.4f}")
    print("===================================")