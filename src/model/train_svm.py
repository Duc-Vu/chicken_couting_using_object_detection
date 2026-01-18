import glob
import numpy as np
import cv2
from sklearn.svm import LinearSVC
import joblib
import os
from ..features.hog import extract_hog

X = []
y = []

for p in glob.glob("data/processed/positive/*.png"):
    img = cv2.imread(p)
    X.append(extract_hog(img))
    y.append(1)

neg_paths = (
    glob.glob("data/processed/negative/*.png") +
    glob.glob("data/processed/hard_negative/*.png")
)

for p in neg_paths:
    img = cv2.imread(p)
    X.append(extract_hog(img))
    y.append(0)

X = np.array(X)
y = np.array(y)

svm = LinearSVC(C=0.01, max_iter=10000)
svm.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(svm, "models/svm_chicken.pkl")

print("Train accuracy:", svm.score(X, y))