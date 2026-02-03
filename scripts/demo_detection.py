import cv2
import joblib
from src.svm_detector.features.hog import extract_hog
from src.svm_detector.inference.sliding_window import run_svm_inference
from src.svm_detector.visualize import draw_detections
from config import config
from src.svm_detector.inference.nms import nms
from datetime import datetime

WINDOW_SIZE = config.WIN_SIZE
STEP = 16
THRESHOLD = 0.2
IS_NMS = True
SAVE = True

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

if __name__ == "__main__":
    svm = joblib.load("models/svm/svm_chicken_v1.5_base.pkl")

    img = cv2.imread(
        "dataset/yolo/test/images/yolov7test4_mp4-55_jpg.rf.3bafea56e4caefeaabb17e600cc9649a.jpg"
    )

    detections = run_svm_inference(
    img=img,
    svm=svm,
    feature_extractor=extract_hog,
    window_size=WINDOW_SIZE,
    step=STEP,
    threshold=THRESHOLD,
    )

    if IS_NMS:
        print("Hello")
        detections = apply_nms(
            detections,
            score_thresh=THRESHOLD,
            nms_thresh=0.3,
        )

    vis = draw_detections(
        img,
        detections,
        WINDOW_SIZE,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imshow("detections", vis)
    cv2.imwrite(f"dataset/svm/save_img/result_{timestamp}.jpg", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
