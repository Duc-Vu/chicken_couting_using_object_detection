import os
from sklearn.metrics import mean_absolute_error

def count_gt_from_yolo(label_path):
    """
    label_path: path tới file .txt YOLO
    return: số object ground truth trong ảnh
    """
    if not os.path.exists(label_path):
        return 0

    with open(label_path, "r") as f:
        lines = f.readlines()

    return len(lines)

def count_pred_boxes(pred_boxes):
    return len(pred_boxes)

def evaluate_counting_mae(
    image_ids,
    label_dir,
    predictions
):
    """
    image_ids : list tên ảnh (không extension) ['img001', 'img002', ...]
    label_dir : thư mục chứa YOLO gt labels
    predictions : dict
        {
          'img001': pred_boxes,
          'img002': pred_boxes,
        }
    """

    gt_counts = []
    pred_counts = []

    for img_id in image_ids:
        label_path = os.path.join(label_dir, img_id + ".txt")

        gt = count_gt_from_yolo(label_path)
        pred = count_pred_boxes(predictions.get(img_id, []))

        gt_counts.append(gt)
        pred_counts.append(pred)

    mae = mean_absolute_error(gt_counts, pred_counts)
    return mae, gt_counts, pred_counts