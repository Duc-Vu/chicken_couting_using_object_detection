from src.svm_detector.features.hog import extract_hog
from src.svm_detector.dataset.hard_negative_mining import hard_negative_mining_video
from config import config
import joblib

def apply_nms(detections, score_thresh=0.0, nms_thresh=0.4):
    boxes = []
    scores = []

    for det in detections:
        print(det)
        x1, y1, x2, y2 = det[:4]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(det[4]))

    indices = nms(boxes, scores, score_thresh, nms_thresh)

    if len(indices) == 0:
        return []

    return [detections[i] for i in indices.flatten()]