import cv2
import numpy as np


def detect_lanes_classical(img, roi_top_frac=0.55, canny_low=80, canny_high=200,
                           hough_threshold=40, min_line_len=30, max_line_gap=20,
                           line_thickness=12):
    """Classical baseline: grayscale -> Gaussian -> Canny -> Hough -> mask.

    Returns a binary mask (uint8, 0/255) the same size as `img` highlighting
    detected lane-line pixels.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)

    roi = np.zeros_like(edges)
    top = int(h * roi_top_frac)
    x_margin = int(w * 0.12)
    poly = np.array([
        [x_margin, h],
        [w - x_margin, h],
        [int(w * 0.5) + 20, top],
        [int(w * 0.5) - 20, top],
    ], np.int32)
    cv2.fillConvexPoly(roi, poly, 255)
    edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    mask = np.zeros((h, w), np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)
    return mask