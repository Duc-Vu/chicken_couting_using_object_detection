from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

def build_svm(C=0.01, max_iter=10000):
    return LinearSVC(C=C, max_iter=max_iter, verbose=1)

def build_sgd(
    alpha=1e-4,
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
    verbose=1,
):
    return SGDClassifier(
        loss="hinge",          # SVM
        alpha=alpha,           # regularization
        max_iter=max_iter,
        class_weight=class_weight,
        early_stopping=True,
        n_iter_no_change=10,
        tol=1e-3,
        random_state=random_state,
        verbose=verbose,
        n_jobs=-1,
    )