"""Standalone point-localization evaluator, not the official KAMP evaluator."""
import numpy as np


def detect(image, threshold: float = 10., min_distance: int = 8):
    """CPU local-contrast baseline. No learned weights, no ground-truth access."""
    image = image.astype(float)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(image, 4, mode="reflect"), (9, 9))
    score = windows.mean(axis=(-2, -1)) - image
    # Image-derived foreground gating rejects the package/air boundary.
    # Threshold 210 is fixture-specific, not calibrated on competition images.
    foreground = (windows < 210).all(axis=(-2, -1))
    y, x = np.where((score >= threshold) & foreground)
    order = np.argsort(-score[y, x], kind="stable")
    predictions = []
    for i in order:
        cx, cy = int(x[i]), int(y[i])
        if all(np.hypot(cx-px, cy-py) >= min_distance for px, py, _ in predictions):
            predictions.append((cx, cy, float(score[cy, cx])))
    return predictions


def evaluate(cases, radius: float = 3.):
    """Confidence-ranked, greedy ONE-TO-ONE point matching within each image.

    AP is the all-points interpolated precision-recall area for the supplied
    candidate list. It is point AP@3px, not COCO box mAP or a contest score.
    Predictions below the detector's candidate threshold are absent from AP.
    """
    if radius <= 0 or not cases:
        raise ValueError("Positive radius and at least one case required")
    total = sum(len(truth) for truth, _ in cases)
    matched = [set() for _ in cases]
    ranked = sorted(((float(s), i, x, y) for i, (_, pred) in enumerate(cases)
                     for x, y, s in pred), key=lambda row: -row[0])
    hits = []
    for _, i, x, y in ranked:
        available = [(float(np.hypot(x-gx, y-gy)), j) for j, (gx, gy) in enumerate(cases[i][0])
                     if j not in matched[i]]
        distance, j = min(available, default=(float("inf"), -1))
        hit = distance <= radius
        if hit:
            matched[i].add(j)
        hits.append(int(hit))
    tp = np.cumsum(hits)
    fp = np.cumsum(1 - np.asarray(hits))
    true_positive = int(sum(hits))
    ap = None
    if total:
        recall = np.r_[0., tp / total, 1.]
        precision = np.r_[0., tp / np.maximum(tp + fp, 1), 0.]
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.sum(np.diff(recall) * precision[1:]))
    return {"images": len(cases), "ground_truth": total,
            "predictions": len(hits), "true_positive": true_positive,
            "recall": true_positive / total if total else None,
            "precision": true_positive / len(hits) if hits else None,
            "unmatched_per_image": (len(hits)-true_positive) / len(cases),
            "point_ap_3px": ap}
