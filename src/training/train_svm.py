import glob
import cv2
import numpy as np


def load_features(img_paths, feature_extractor, label):
    X, y = [], []

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        X.append(feature_extractor(img))
        y.append(label)

    return X, y


def train_svm_classifier(
    pos_dir,
    neg_dirs,
    feature_extractor,
    svm,
):
    X, y = [], []

    pos_paths = glob.glob(f"{pos_dir}/*.png")
    Xp, yp = load_features(pos_paths, feature_extractor, 1)
    X.extend(Xp)
    y.extend(yp)

    for neg_dir in neg_dirs:
        neg_paths = glob.glob(f"{neg_dir}/*.png")
        Xn, yn = load_features(neg_paths, feature_extractor, 0)
        X.extend(Xn)
        y.extend(yn)

    X = np.array(X)
    y = np.array(y)

    svm.fit(X, y)

    return svm, X, y
