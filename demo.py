"""Run: python demo.py --output runs/demo --seed 2026"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from augmentation import fixture, extract_bank, augment, yolo_labels
from evaluation import detect, evaluate


def save_sample(root, split, name, sample):
    for kind in ("images", "labels"):
        (root / kind / split).mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample.image).save(root / "images" / split / f"{name}.png")
    (root / "labels" / split / f"{name}.txt").write_text(yolo_labels(sample), encoding="utf-8")


def preview(before, after, predictions, path):
    canvas = Image.new("RGB", (1000, 365), "#0b1220")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "KAMP | Equipment-conditioned residual augmentation", fill="white", font_size=23)
    for i, (sample, title) in enumerate([(before, "1. Procedural product"),
                                       (after, "2. Residual augmentation"),
                                       (after, "3. CPU detection baseline")]):
        im = Image.fromarray(sample.image).convert("RGB").resize((320, 256))
        d = ImageDraw.Draw(im)
        if i > 0:
            for x, y in sample.centers:
                d.rectangle((2*x-8, 2*y-8, 2*x+8, 2*y+8), outline="#42e4a5", width=2)
        if i == 2:
            for x, y, _ in predictions:
                d.ellipse((2*x-11, 2*y-11, 2*x+11, 2*y+11), outline="#ffab48", width=2)
        canvas.paste(im, (10 + 330*i, 74))
        draw.text((10 + 330*i, 48), title, fill="#a9c5dd", font_size=17)
    draw.text((15, 339), "Procedural fixtures only | green: synthetic labels | orange: predictions", fill="#a9c5dd", font_size=16)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    donors = [fixture(args.seed+i, eq, 2) for i, eq in enumerate(["A", "B"] * 6)]
    banks = extract_bank(donors)
    for i, donor in enumerate(donors):
        augmented = augment(donor, banks, args.seed+100+i, count=1)
        save_sample(root, "train", f"train_{i:03}", augmented)
    # Independent fixture seeds, never included in the training donor banks.
    all_cases = []
    per_equipment = {}
    for eq in ("A", "B"):
        cases = []
        for i in range(12):
            sample = fixture(args.seed+1000+i+(100 if eq == "B" else 0), eq, i % 3)
            predictions = detect(sample.image)
            cases.append((sample.centers, predictions))
            save_sample(root, "val", f"val_{eq}_{i:03}", sample)
        per_equipment[eq] = evaluate(cases)
        all_cases.extend(cases)
    clean = fixture(args.seed+5000, "A")
    augmented = augment(clean, banks, args.seed+6000, 2)
    preview(clean, augmented, detect(augmented.image), root / "preview.png")
    result = {"scope": "procedural fixtures; not KAMP benchmark results",
              "seed": args.seed, "detector": "CPU local contrast; no learned weights",
              "matching": "one-to-one point distance <= 3 pixels",
              "candidate_threshold": 10.0, "train_images": len(donors),
              "donor_bank_sizes": {k: len(v) for k, v in banks.items()},
              "aggregate": evaluate(all_cases), "by_equipment": per_equipment}
    (root / "metrics.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    # Relative path keeps the generated dataset portable; pass its resolved YAML to YOLO.
    (root / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: foreign_object\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Preview and YOLO-format dataset: {root.resolve()}")


if __name__ == "__main__":
    main()
