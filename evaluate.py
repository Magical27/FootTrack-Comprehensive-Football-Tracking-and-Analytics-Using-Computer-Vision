import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO validation and report precision/recall/mAP metrics."
    )
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to trained YOLO weights (default: models/best.pt)",
    )
    parser.add_argument(
        "--data",
        default="dataset/data.yaml",
        help="Path to dataset yaml (default: dataset/data.yaml)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--device",
        default=None,
        help='Device for evaluation, e.g. "0", "cpu" (default: auto)',
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Ask ultralytics to also save COCO-style prediction json",
    )
    parser.add_argument(
        "--output",
        default="evaluation",
        help="Directory to store metrics report (default: evaluation)",
    )
    return parser.parse_args()


def pick_metric(results_dict, key_options):
    for key in key_options:
        if key in results_dict:
            return float(results_dict[key])
    return None


def main():
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=args.save_json,
        verbose=True,
    )

    results_dict = metrics.results_dict if hasattr(metrics, "results_dict") else {}

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": {
            "precision": pick_metric(results_dict, ["metrics/precision(B)", "metrics/precision"]),
            "recall": pick_metric(results_dict, ["metrics/recall(B)", "metrics/recall"]),
            "mAP50": pick_metric(results_dict, ["metrics/mAP50(B)", "metrics/mAP50"]),
            "mAP50-95": pick_metric(results_dict, ["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
        },
        "raw_results_dict": results_dict,
    }

    print("\nEvaluation summary")
    print("------------------")
    print(f"Model      : {summary['model']}")
    print(f"Data       : {summary['data']}")
    print(f"Split      : {summary['split']}")
    print(f"Precision  : {summary['metrics']['precision']}")
    print(f"Recall     : {summary['metrics']['recall']}")
    print(f"mAP@0.50   : {summary['metrics']['mAP50']}")
    print(f"mAP@0.50:95: {summary['metrics']['mAP50-95']}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"metrics_{args.split}_{stamp}.json"
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved metrics report: {output_file}")


if __name__ == "__main__":
    main()
