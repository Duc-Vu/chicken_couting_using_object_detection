import os
import json
import cv2


def load_coco(coco_path):
    with open(coco_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_map(coco):
    return {
        img["id"]: img["file_name"]
        for img in coco["images"]
    }


def crop_and_resize(img, bbox, out_size):
    x, y, w, h = map(int, bbox)
    crop = img[y:y + h, x:x + w]

    if crop.size == 0:
        return None

    return cv2.resize(crop, out_size)


def generate_positive_patches(
    coco_path,
    image_dir,
    output_dir,
    out_size=(64, 64),
):
    os.makedirs(output_dir, exist_ok=True)

    coco = load_coco(coco_path)
    image_map = build_image_map(coco)

    pos_id = 0

    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        img_path = os.path.join(image_dir, image_map[image_id])

        img = cv2.imread(img_path)
        if img is None:
            continue

        patch = crop_and_resize(
            img,
            ann["bbox"],
            out_size
        )

        if patch is None:
            continue

        out_path = os.path.join(
            output_dir,
            f"pos_{pos_id}.png"
        )
        cv2.imwrite(out_path, patch)
        print(f"Positive Crop {pos_id}")
        pos_id += 1

    return pos_id
