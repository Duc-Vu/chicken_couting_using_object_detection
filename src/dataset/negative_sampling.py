import os
import json
import random
import cv2

def load_coco(coco_path):
    with open(coco_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_image_map(coco):
    return {
        img["id"]: img["file_name"]
        for img in coco["images"]
    }


def get_bboxes_for_image(coco, image_id):
    return [
        ann["bbox"]
        for ann in coco["annotations"]
        if ann["image_id"] == image_id
    ]


def is_valid_negative(x, y, patch_size, bboxes):
    for bx, by, bw, bh in bboxes:
        if not (
            x + patch_size < bx or
            x > bx + bw or
            y + patch_size < by or
            y > by + bh
        ):
            return False
    return True


def generate_negative_patches(
    coco_path,
    image_dir,
    output_dir,
    patch_size=64,
    samples_per_image=35,
    var_thresh=15,
    seed=50,
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    coco = load_coco(coco_path)

    neg_id = 0

    for img_info in coco["images"]:
        img_path = os.path.join(image_dir, img_info["file_name"])
        img = cv2.imread(img_path)

        if img is None:
            continue

        H, W = img.shape[:2]
        bboxes = get_bboxes_for_image(coco, img_info["id"])

        if W < patch_size or H < patch_size:
            continue

        for _ in range(samples_per_image):
            x = random.randint(0, W - patch_size)
            y = random.randint(0, H - patch_size)

            if not is_valid_negative(x, y, patch_size, bboxes):
                continue

            patch = img[y:y + patch_size, x:x + patch_size]

            out_path = os.path.join(
                output_dir,
                f"neg_{neg_id}.png"
            )
            cv2.imwrite(out_path, patch)
            print(f"Negative Crop {neg_id}")
            neg_id += 1
