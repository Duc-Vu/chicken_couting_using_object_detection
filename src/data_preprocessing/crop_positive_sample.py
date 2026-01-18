import json
import cv2
import os

train_coco_path = "data/raw/train/annotations/coco.json"

with open(train_coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

image_map = {
    img["id"]: img["file_name"]
    for img in coco["images"]
}

os.makedirs("data/processed/positive", exist_ok=True)

pos_id = 0

for ann in coco["annotations"]:
    image_id = ann["image_id"]
    x, y, w, h = map(int, ann["bbox"])

    img_path = os.path.join("data/raw/train/images", image_map[image_id])
    img = cv2.imread(img_path)

    if img is None:
        continue

    crop = img[y:y+h, x:x+w]

    if crop.size == 0:
        continue

    crop = cv2.resize(crop, (64, 64))

    cv2.imwrite(f"data/processed/positive/pos_{pos_id}.png", crop)
    print(f"Positive Crop {pos_id}")
    pos_id += 1

