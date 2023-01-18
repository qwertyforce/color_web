import cv2
import numpy as np

def get_color_features(query_image):
    query_hist_combined = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    query_hist_combined = query_hist_combined.flatten()
    query_hist_combined = query_hist_combined*10000000
    query_hist_combined = np.divide(query_hist_combined, query_image.shape[0]*query_image.shape[1], dtype=np.float32)
    return query_hist_combined