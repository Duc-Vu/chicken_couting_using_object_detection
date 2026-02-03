import cv2

def nms(boxes, scores, score_threshold, nms_threshold=0.3):
    indices = cv2.dnn.NMSBoxes(
    boxes,        
    scores,       
    score_threshold=score_threshold,
    nms_threshold=nms_threshold
    )
    return indices