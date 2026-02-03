import os
import joblib
import glob

from src.svm_detector.features.hog import extract_hog
from src.svm_detector.model.svm import build_svm
from src.svm_detector.training.train import train_svm_classifier, load_features
from src.svm_detector.evaluate import evaluate_classifier
from src.svm_detector.utils.logger import setup_file_logger
from config import config


TRAIN_POS = "dataset/svm/train/positive"
TRAIN_NEG = [
    "dataset/svm/train/negative",
    "dataset/svm/train/hard_negative",
]

VALID_POS = "dataset/svm/valid/positive"
VALID_NEG = "dataset/svm/valid/negative"
LOG_PATH = "logs/svm"
VERSION = 1.5
SUB_VER = "base"

def glob_images(dir_path):
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    return paths

if __name__ == "__main__":
    logger, log_path = setup_file_logger(
        log_dir=LOG_PATH,
        name=f"svm_hog_v{VERSION}_{SUB_VER}"
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

    Xp, yp = load_features(glob_images(VALID_POS), extract_hog, 1)
    Xn, yn = load_features(glob_images(VALID_NEG), extract_hog, 0)

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
    model_path = f"models/svm/svm_chicken_v{VERSION}_{SUB_VER}.pkl"
    joblib.dump(svm, model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info("===== EXPERIMENT END =====")

    print(f"Log saved to: {log_path}")
