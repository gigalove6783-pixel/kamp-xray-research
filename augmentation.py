"""Equipment-conditioned residual augmentation. Coordinates are (x, y)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    mask: np.ndarray
    centers: list[tuple[int, int]]
    equipment: str


Y, X = np.mgrid[-4:5, -4:5]
RADIUS = np.hypot(X, Y)
CORE = RADIUS <= 2


def fixture(seed: int, equipment: str, defects: int = 0) -> Sample:
    """Generate a synthetic product image with labeled dark spots."""
    if equipment not in {"A", "B"}:
        raise ValueError("Unknown fixture equipment")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:128, :160]
    mask = ((x - 80) / 64) ** 2 + ((y - 64) / 49) ** 2 < 1
    # Background texture and a darker seam make the localization non-trivial.
    body = 158 + 5 * np.sin(x / 11) + 3 * np.cos(y / 9)
    body -= 17 * np.exp(-((x - 76) / 4) ** 2)
    image = np.where(mask, body, 235.) + rng.normal(0, 1.4, x.shape)
    candidates = np.argwhere(mask & (x > 30) & (x < 130) & (y > 30) & (y < 98))
    centers = []
    for _ in range(defects):
        for _ in range(100):
            cy, cx = candidates[rng.integers(len(candidates))]
            if all(np.hypot(cx - px, cy - py) >= 18 for px, py in centers):
                break
        else:
            raise ValueError("Cannot place requested fixture defects")
        amplitude = rng.uniform(33, 55) if equipment == "A" else rng.uniform(20, 37)
        width = .8 if equipment == "A" else 1.1
        residual = -amplitude * np.exp(-(X ** 2 + Y ** 2) / (2 * width ** 2)) * CORE
        image[cy-4:cy+5, cx-4:cx+5] += residual
        centers.append((int(cx), int(cy)))
    return Sample(np.clip(np.rint(image), 0, 255).astype(np.uint8), mask, centers, equipment)


def extract_bank(samples: list[Sample]) -> dict[str, list[np.ndarray]]:
    """Extract dark cores using an annular background estimate.

    Pass training samples only; the split is managed by the caller.
    """
    banks = {}
    for sample in samples:
        y, x = np.indices(sample.image.shape)
        for cx, cy in sample.centers:
            if not (4 <= cx < x.shape[1]-4 and 4 <= cy < x.shape[0]-4):
                continue
            radius = np.hypot(x-cx, y-cy)
            ring = (radius >= 7) & (radius <= 10) & sample.mask
            for px, py in sample.centers:
                ring &= np.hypot(x-px, y-py) > 4
            if ring.sum() < 8:
                continue
            background = np.median(sample.image[ring])
            patch = sample.image[cy-4:cy+5, cx-4:cx+5].astype(float)
            residual = np.minimum(patch - background, 0)
            residual -= residual[(RADIUS >= 3) & (RADIUS <= 4)].mean()
            residual[(residual > -10) | ~CORE] = 0
            if np.any(residual < 0):
                banks.setdefault(sample.equipment, []).append(residual)
    return banks


def augment(sample: Sample, banks: dict, seed: int, count: int = 2) -> Sample:
    """Insert same-equipment residuals within the product mask, 20px apart."""
    if count < 0:
        raise ValueError("count must be non-negative")
    bank = banks.get(sample.equipment)
    if not bank:
        raise ValueError(f"No training donors for equipment {sample.equipment}")
    rng = np.random.default_rng(seed)
    windows = np.lib.stride_tricks.sliding_window_view(sample.mask, (9, 9))
    candidates = np.argwhere(windows.all(axis=(-2, -1))) + 4
    rng.shuffle(candidates)
    centers = list(sample.centers)
    image = sample.image.astype(float).copy()
    inserted = 0
    for cy, cx in candidates:
        if inserted == count:
            break
        if any(np.hypot(cx-px, cy-py) < 20 for px, py in centers):
            continue
        image[cy-4:cy+5, cx-4:cx+5] += bank[rng.integers(len(bank))]
        centers.append((int(cx), int(cy)))
        inserted += 1
    if inserted != count:
        raise ValueError("Insufficient valid insertion locations")
    return Sample(np.clip(np.rint(image), 0, 255).astype(np.uint8),
                  sample.mask.copy(), centers, sample.equipment)


def yolo_labels(sample: Sample, box_size: int = 7) -> str:
    """Export fixed-size fixture boxes in YOLO format."""
    height, width = sample.image.shape
    return "".join(f"0 {(x+.5)/width:.6f} {(y+.5)/height:.6f} "
                   f"{box_size/width:.6f} {box_size/height:.6f}\n" for x, y in sample.centers)
