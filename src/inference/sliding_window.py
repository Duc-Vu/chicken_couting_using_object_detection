import cv2
import numpy as np


def sliding_window(img, window_size, step):
    H, W = img.shape[:2]

    for y in range(0, H - window_size, step):
        for x in range(0, W - window_size, step):
            yield x, y, img[y:y + window_size, x:x + window_size]


def run_svm_inference(
    img,
    svm,
    feature_extractor,
    window_size=64,
    step=16,
    threshold=0.0,
):
    detections = []

    for x, y, patch in sliding_window(img, window_size, step):
        feat = feature_extractor(patch).reshape(1, -1)
        score = svm.decision_function(feat)[0]

        if score > threshold:
            detections.append((x, y, score))

    return detections
