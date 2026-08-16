import numpy as np


def _confusion(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(pred.sum()) - tp
    fn = int(gt.sum()) - tp
    return tp, fp, fn


def iou(pred, gt):
    tp, fp, fn = _confusion(pred, gt)
    union = tp + fp + fn
    return tp / union if union else float("nan")


def f1_score(pred, gt):
    tp, fp, fn = _confusion(pred, gt)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std())