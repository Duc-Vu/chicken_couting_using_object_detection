import os
from math import ceil

from src.dataset.negative_sampling import generate_negative_patches
from src.dataset.positive_sampling import generate_positive_patches


NEG_RATIO = 3


def count_images(image_dir):
    return len([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])


def build_split(
    coco_path,
    image_dir,
    out_pos_dir,
    out_neg_dir,
    split_name,
):
    print(f"\n===== BUILD {split_name.upper()} =====")

    num_pos = generate_positive_patches(
        coco_path=coco_path,
        image_dir=image_dir,
        output_dir=out_pos_dir,
    )

    num_imgs = count_images(image_dir)

    samples_per_image = ceil(
        NEG_RATIO * num_pos / num_imgs
    )

    print(f"{split_name}:")
    print(f"  images = {num_imgs}")
    print(f"  positives = {num_pos}")
    print(f"  samples_per_image (neg) = {samples_per_image}")

    num_neg = generate_negative_patches(
        coco_path=coco_path,
        image_dir=image_dir,
        output_dir=out_neg_dir,
        samples_per_image=samples_per_image,
    )

    print(f"  negatives = {num_neg}")
    print(f"  ratio neg/pos ≈ {num_neg / num_pos:.2f}")


if __name__ == "__main__":

    build_split(
        coco_path="data/raw/train/coco.json",
        image_dir="data/raw/train",
        out_pos_dir="data/processed/train/positive",
        out_neg_dir="data/processed/train/negative",
        split_name="train",
    )

    build_split(
        coco_path="data/raw/test/coco.json",
        image_dir="data/raw/test",
        out_pos_dir="data/processed/test/positive",
        out_neg_dir="data/processed/test/negative",
        split_name="test",
    )
