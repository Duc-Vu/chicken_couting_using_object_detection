import cv2


def draw_detections(
    img,
    detections,
    color=(0, 0, 255),
):
    vis = img.copy()

    for det in detections:
        x1, y1, x2, y2 = det[:4]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        score = det[4]
        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )
        cv2.putText(
            vis,
            f"{score:.2f}",
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    return vis
