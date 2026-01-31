import cv2
import joblib
from src.svm_detector.features.hog import extract_hog
from src.svm_detector.inference.sliding_window import run_svm_inference
from svm_detector.visualize import draw_detections


WINDOW_SIZE = 64
STEP = 16
THRESHOLD = 0.0


if __name__ == "__main__":
    svm = joblib.load("models/svm_chicken.pkl")

    img = cv2.imread(
        "data/raw/test/bc688d39308b1adc_jpg.rf.42422b119647f1764392fbecb5934e71.jpg"
    )

    detections = run_svm_inference(
        img=img,
        svm=svm,
        feature_extractor=extract_hog,
        window_size=WINDOW_SIZE,
        step=STEP,
        threshold=THRESHOLD,
    )

    vis = draw_detections(
        img,
        detections,
        WINDOW_SIZE,
    )

    cv2.imshow("detections", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
