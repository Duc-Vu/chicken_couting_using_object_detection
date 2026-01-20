from src.data.negative_sampling import generate_negative_patches
from src.data.positive_sampling import generate_positive_patches

COCO_TRAIN_PATH = "data/raw/train/coco.json" 
IMAGE_TRAIN_PATH = "data/raw/train"
OUTPUT_NEG_TRAIN_PATH = "data/processed/train/negative"
OUTPUT_POS_TRAIN_PATH = "data/processed/train/positive"

COCO_TEST_PATH = "data/raw/test/coco.json" 
IMAGE_TEST_PATH = "data/raw/test"
OUTPUT_NEG_TEST_PATH = "data/processed/test/negative"
OUTPUT_POS_TEST_PATH = "data/processed/test/positive"

if __name__ == "__main__":
    generate_negative_patches(coco_path=COCO_TRAIN_PATH, image_dir=IMAGE_TRAIN_PATH,
                              output_dir=OUTPUT_NEG_TRAIN_PATH)
    generate_positive_patches(coco_path=COCO_TRAIN_PATH, image_dir=IMAGE_TRAIN_PATH,
                              output_dir=OUTPUT_POS_TRAIN_PATH)
    
    generate_negative_patches(coco_path=COCO_TEST_PATH, image_dir=IMAGE_TEST_PATH,
                              output_dir=OUTPUT_NEG_TEST_PATH)
    generate_positive_patches(coco_path=COCO_TRAIN_PATH, image_dir=IMAGE_TRAIN_PATH,
                              output_dir=OUTPUT_POS_TEST_PATH)