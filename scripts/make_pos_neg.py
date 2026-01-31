import os
from math import ceil

from src.dataset.negative_sampling import generate_negative
from src.dataset.positive_sampling import generate_positive


NEG_RATIO = 3


def count_images(image_dir):
    return len([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])


def build_split(
    image_dir,
    label_dir,
    out_pos_dir,
    out_neg_dir,
    split_name,
):
    print(f"\n===== BUILD {split_name.upper()} =====")

    # ---- POSITIVE ----
    num_pos = generate_positive(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=out_pos_dir,
    )

    num_imgs = count_images(image_dir)

    samples_per_image = ceil(
        NEG_RATIO * num_pos / max(num_imgs, 1)
    )

    print(f"{split_name}:")
    print(f"  images = {num_imgs}")
    print(f"  positives = {num_pos}")
    print(f"  samples_per_image (neg) = {samples_per_image}")

    # ---- NEGATIVE ----
    num_neg = generate_negative(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=out_neg_dir,
        samples_per_image=samples_per_image,
    )
    print(f"{split_name}:")
    print(f"  images = {num_imgs}")
    print(f"  positives = {num_pos}")
    print(f"  samples_per_image (neg) = {samples_per_image}")
    print(f"  negatives = {num_neg}")
    print(f"  ratio neg/pos ≈ {num_neg / max(num_pos, 1):.2f}")

if __name__ == "__main__":

    build_split(
        image_dir="dataset/yolo/train/images",
        label_dir="dataset/yolo/train/labels",
        out_pos_dir="dataset/svm/train/positive",
        out_neg_dir="dataset/svm/train/negative",
        split_name="train",
    )

    build_split(
        image_dir="dataset/yolo/test/images",
        label_dir="dataset/yolo/test/labels",
        out_pos_dir="dataset/svm/valid/positive",
        out_neg_dir="dataset/svm/valid/negative",
        split_name="valid",
    )
