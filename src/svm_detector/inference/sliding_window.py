import cv2
import numpy as np


def sliding_window(img, window_size, step):
    H, W = img.shape[:2]

    for y in range(0, H - window_size[1], step):
        for x in range(0, W - window_size[0], step):
            yield x, y, img[y:y + window_size[1], x:x + window_size[0]]


def run_svm_inference(
    img,
    svm,
    feature_extractor,
    window_size=(64, 128),   # (W, H) cho gà
    step=16,
    threshold=0.0,
    nms_thresh=0.4
):
    detections = []

    W, H = window_size

    for x, y, patch in sliding_window(img, window_size, step):
        # bảo vệ: patch phải đúng size
        if patch.shape[1] != W or patch.shape[0] != H:
            continue

        feat = feature_extractor(patch).reshape(1, -1)
        score = svm.decision_function(feat)[0]

        if score > threshold:
            # box dạng (x1, y1, x2, y2)
            detections.append([
                x,
                y,
                x + W,
                y + H,
                score
            ])

    if len(detections) == 0:
        return []

    detections = np.array(detections)

    boxes  = detections[:, :4]
    scores = detections[:, 4]

    # dùng NMS có sẵn của OpenCV
    indices = cv2.dnn.NMSBoxes(
        bboxes=[(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])) for b in boxes],
        scores=scores.tolist(),
        score_threshold=threshold,
        nms_threshold=nms_thresh
    )

    final_dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_dets.append(detections[i])

    return final_dets
