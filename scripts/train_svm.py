import os
import joblib
import glob

from src.features.hog import extract_hog
from src.models.svm import build_svm
from src.training.train_svm import train_svm_classifier, load_features
from src.evaluation.classification import evaluate_classifier
from src.utils.logger import setup_file_logger



TRAIN_POS = "data/processed/train/positive"
TRAIN_NEG = [
    "data/processed/train/negative",
    "data/processed/train/hard_negative",
]

TEST_POS = "data/processed/test/positive"
TEST_NEG = "data/processed/test/negative"

def glob_images(dir_path):
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    return paths

if __name__ == "__main__":
    logger, log_path = setup_file_logger(
        name="svm_hog"
    )

    logger.info("===== EXPERIMENT START =====")

    # ===== CONFIG =====
    C = 0.01
    MAX_ITER = 10000

    logger.info("Config:")
    logger.info(f"  Model: LinearSVC")
    logger.info(f"  Feature: HOG")
    logger.info(f"  C={C}, max_iter={MAX_ITER}")

    # ===== TRAIN =====
    svm = build_svm(C=C, max_iter=MAX_ITER)

    svm, X_train, y_train = train_svm_classifier(
        pos_dir=TRAIN_POS,
        neg_dirs=TRAIN_NEG,
        feature_extractor=extract_hog,
        svm=svm,
    )

    logger.info(f"Train samples: {len(y_train)}")
    logger.info(f"Train pos: {sum(y_train)} | neg: {len(y_train)-sum(y_train)}")

    train_metrics = evaluate_classifier(
        svm, X_train, y_train
    )
    for k, v in train_metrics.items():
        logger.info(f"Train {k}: {v:.4f}")

    # ===== TEST =====
    X_test, y_test = [], []

    Xp, yp = load_features(glob_images(TEST_POS), extract_hog, 1)
    Xn, yn = load_features(glob_images(TEST_NEG), extract_hog, 0)

    X_test.extend(Xp)
    y_test.extend(yp)
    X_test.extend(Xn)
    y_test.extend(yn)

    test_metrics = evaluate_classifier(
        svm, X_test, y_test
    )

    logger.info(f"Test samples: {len(y_test)}")
    logger.info(f"Test pos: {sum(y_test)} | neg: {len(y_test)-sum(y_test)}")

    for k, v in test_metrics.items():
        logger.info(f"Test {k}: {v:.4f}")

    # ===== SAVE =====
    os.makedirs("models", exist_ok=True)
    model_path = "models/svm_chicken.pkl"
    joblib.dump(svm, model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info("===== EXPERIMENT END =====")

    print(f"Log saved to: {log_path}")
