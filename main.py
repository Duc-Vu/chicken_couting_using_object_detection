import cv2
import joblib
import numpy as np
from src.features.hog import extract_hog

WINDOW_SIZE = 64
STEP = 16      
THRESHOLD = 0.0   

svm = joblib.load("models/svm_chicken.pkl")

img = cv2.imread("data/raw/train/images/before_mp4-12_jpg.rf.682b991d442b0eaecff606bbfd3211c7.jpg")
H, W = img.shape[:2]

detections = []

for y in range(0, H - WINDOW_SIZE, STEP):
    for x in range(0, W - WINDOW_SIZE, STEP):
        patch = img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
        feat = extract_hog(patch).reshape(1, -1)

        score = svm.decision_function(feat)[0]

        if score > THRESHOLD:
            detections.append((x, y, score))

for x, y, score in detections:
    cv2.rectangle(
        img,
        (x, y),
        (x + WINDOW_SIZE, y + WINDOW_SIZE),
        (0, 0, 255),
        1
    )
    cv2.putText(
        img,
        f"{score:.2f}",
        (x, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 255, 0),
        1
    )

cv2.imshow("detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()