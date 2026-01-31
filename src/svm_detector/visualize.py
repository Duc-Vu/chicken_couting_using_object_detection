import cv2


def draw_detections(
    img,
    detections,
    window_size,
    color=(0, 0, 255),
):
    vis = img.copy()

    for x, y, score in detections:
        cv2.rectangle(
            vis,
            (x, y),
            (x + window_size, y + window_size),
            color,
            1,
        )
        cv2.putText(
            vis,
            f"{score:.2f}",
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
        )

    return vis
