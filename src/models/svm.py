from sklearn.svm import LinearSVC

def build_svm(C=0.01, max_iter=10000):
    return LinearSVC(C=C, max_iter=max_iter)