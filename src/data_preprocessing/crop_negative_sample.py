import random
import json
import cv2
import os

random.seed(50)

train_coco_path = "data/raw/train/annotations/coco.json"

with open(train_coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

image_map = {
    img["id"]: img["file_name"]
    for img in coco["images"]
}

def is_blank_patch(patch, var_thresh=15):

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return gray.var() < var_thresh


os.makedirs("data/processed/negative", exist_ok=True)

neg_id = 0
PATCH_SIZE = 64

for img_info in coco["images"]:
    img_path = os.path.join("data/raw/train/images", img_info["file_name"])
    img = cv2.imread(img_path)

    if img is None:
        continue

    H, W = img.shape[:2]

    bboxes = [
        ann["bbox"]
        for ann in coco["annotations"]
        if ann["image_id"] == img_info["id"]
    ]

    for _ in range(35):
        x = random.randint(0, W - PATCH_SIZE)
        y = random.randint(0, H - PATCH_SIZE)

        ok = True
        for bx, by, bw, bh in bboxes:
            if not (x + PATCH_SIZE < bx or
                    x > bx + bw or
                    y + PATCH_SIZE < by or
                    y > by + bh):
                ok = False
                break

        if ok:
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            
            if is_blank_patch(patch):
                continue
            
            cv2.imwrite(f"data/processed/negative/neg_{neg_id}.png", patch)
            print(f"Negative Crop {neg_id}")
            neg_id += 1