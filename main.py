import cv2
import joblib
from src.features.hog import extract_hog
from src.inference.nms import nms
WINDOW = 64
STEP = 16
SCORE_THRESH = 0.5

svm = joblib.load("models/svm_chicken.pkl")
cap = cv2.VideoCapture("data/test/test_chicken.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    boxes, scores = [], []

    for y in range(0, H - WINDOW, STEP):
        for x in range(0, W - WINDOW, STEP):
            patch = frame[y:y+WINDOW, x:x+WINDOW]
            feat = extract_hog(patch).reshape(1, -1)
            score = svm.decision_function(feat)[0]

            if score > SCORE_THRESH:
                boxes.append([x, y, x+WINDOW, y+WINDOW])
                scores.append(score)

    keep = nms(boxes, scores, iou_thresh=0.3)
    final_boxes = [boxes[i] for i in keep]

    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.putText(
        frame,
        f"Chicken count: {len(final_boxes)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("Chicken Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
