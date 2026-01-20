import os
import cv2


def overlap_with_any_gt(x, y, w, h, gt_bboxes):
    for bx, by, bw, bh in gt_bboxes:
        if not (x + w < bx or x > bx + bw or y + h < by or y > by + bh):
            return True
    return False


def mine_hard_negatives(
    img,
    svm,
    feature_extractor,
    gt_bboxes,
    output_dir,
    start_id,
    window_size=64,
    step=16,
    score_thresh=0.0,
    max_per_image=10,
):
    H, W = img.shape[:2]
    hneg_id = start_id
    count = 0

    for y in range(0, H - window_size, step):
        for x in range(0, W - window_size, step):
            patch = img[y:y + window_size, x:x + window_size]

            feat = feature_extractor(patch).reshape(1, -1)
            score = svm.decision_function(feat)[0]

            if score <= score_thresh:
                continue

            if overlap_with_any_gt(x, y, window_size, window_size, gt_bboxes):
                continue

            out_path = os.path.join(
                output_dir, f"hneg_{hneg_id}.png"
            )
            cv2.imwrite(out_path, patch)
            print(f"HardNeg {hneg_id} | score={score:.2f}")

            hneg_id += 1
            count += 1

            if count >= max_per_image:
                return hneg_id

    return hneg_id
