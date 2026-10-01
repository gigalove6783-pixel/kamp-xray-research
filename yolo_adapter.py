"""Train or predict with a local Ultralytics checkpoint."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("Supply an existing local checkpoint; automatic downloads are not used")
    if args.mode == "train" and (args.data is None or not args.data.is_file()):
        parser.error("train requires --data pointing to an existing dataset.yaml")
    if args.mode == "predict" and (args.source is None or not args.source.exists()):
        parser.error("predict requires --source pointing to existing images")
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()))
    if args.mode == "train":
        # Explicit root avoids dependence on Ultralytics' global datasets directory.
        import yaml
        data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
        data["path"] = str(args.data.resolve().parent)
        resolved = args.data.with_name("dataset.resolved.yaml")
        resolved.write_text(yaml.safe_dump(data), encoding="utf-8")
        model.train(data=str(resolved.resolve()), epochs=args.epochs, imgsz=160,
                    batch=4, device=args.device, seed=2026, workers=0,
                    project="runs/yolo", name="train", plots=False)
    else:
        for _ in model.predict(source=str(args.source.resolve()), device=args.device,
                               stream=True, save=True, save_txt=True, save_conf=True,
                               project="runs/yolo", name="predict"):
            pass


if __name__ == "__main__":
    main()
