import cv2
import numpy as np


def _line_params(x1, y1, x2, y2):
    """Fit x = a*y + b (stable for near-vertical lanes). Returns None if horizontal."""
    dy = y2 - y1
    if abs(dy) < 5:
        return None
    a = (x2 - x1) / dy
    if abs(a) > 1.6:
        return None
    b = x1 - a * y1
    return a, b


def _cluster_lanes(lines, angle_tol=0.3, intercept_tol=70):
    """Greedily merge near-vertical segments into lane lines."""
    params = [_line_params(*ln) for ln in lines]
    params = [p for p in params if p is not None]
    clusters = []
    for a, b in params:
        placed = False
        for cl in clusters:
            if abs(a - cl["a"]) < angle_tol and abs(b - cl["b"]) < intercept_tol:
                cl["a"] = (cl["a"] * cl["n"] + a) / (cl["n"] + 1)
                cl["b"] = (cl["b"] * cl["n"] + b) / (cl["n"] + 1)
                cl["n"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"a": a, "b": b, "n": 1})
    clusters.sort(key=lambda c: c["n"], reverse=True)
    return clusters


def detect_lanes_classical(img, near_top_frac=0.78, canny_low=40, canny_high=120,
                           hough_threshold=25, min_line_len=20, max_line_gap=15,
                           line_thickness=12, max_lanes=4):
    """Classical baseline: grayscale -> Gaussian -> Canny -> Hough, near-field band.

    TuSimple lane markings are often faded and lose all luminance contrast with the
    road in the far field, so detection runs in the bottom band of the frame (where
    the paint is still visible), near-vertical segments are clustered by slope and
    intercept, and the strongest clusters are extrapolated into full lane lines.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high)

    y0 = int(h * near_top_frac)
    roi = np.zeros_like(edges)
    cv2.rectangle(roi, (int(w * 0.05), y0), (int(w * 0.95), h - 1), 255, -1)
    edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    mask = np.zeros((h, w), np.uint8)
    if lines is None:
        return mask

    for lane in _cluster_lanes(lines.reshape(-1, 4))[:max_lanes]:
        a, b = lane["a"], lane["b"]
        y_top = int(y0 * 0.8)
        y_bot = h - 1
        cv2.line(mask, (int(a * y_top + b), y_top), (int(a * y_bot + b), y_bot), 255, line_thickness)
    return mask