from skimage.feature import hog
import cv2
from config import config

def extract_hog(img, win_size=config.WIN_SIZE):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) 

    hog = cv2.HOGDescriptor(
        _winSize=win_size,      
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    feat = hog.compute(gray)
    return feat

