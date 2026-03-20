#!/usr/bin/env python3
"""
Package an ensemble submission: multiple ONNX models + ensemble run.py.

Usage:
  python package_ensemble.py --runs v63_x_full v62_rtdetr_full
  python package_ensemble.py --runs v63_x_full v62_rtdetr_full --imgsz 1600 1600
"""
import argparse
import shutil
import zipfile
from pathlib import Path

WORKDIR = Path("/cluster/home/ksv023/NM_AI_2026/task1")
MAX_SIZE_MB = 420


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None,
                        help="ONNX filenames (default: model1.onnx, model2.onnx, ...)")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None)
    parser.add_argument("--half", nargs="+", type=int, default=None,
                        help="Per-model FP16 flags: 0=FP32, 1=FP16 (default: all FP32)")
    parser.add_argument("--out", default="submission_ensemble")
    args = parser.parse_args()

    output_dir = WORKDIR / args.out
    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging = output_dir / "staging"
    staging.mkdir(parents=True)

    print(f"Packaging ensemble: {args.runs}")

    from ultralytics import YOLO

    total_onnx_mb = 0
    for i, run_name in enumerate(args.runs):
        weights = WORKDIR / "runs" / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"  ERROR: {weights} not found")
            return

        # Detect imgsz from args.yaml
        if args.imgsz and i < len(args.imgsz):
            imgsz = args.imgsz[i]
        else:
            import yaml
            args_yaml = WORKDIR / "runs" / run_name / "args.yaml"
            if args_yaml.exists():
                with open(args_yaml) as f:
                    imgsz = yaml.safe_load(f).get("imgsz", 1280)
            else:
                imgsz = 1280

        use_half = bool(args.half[i]) if args.half and i < len(args.half) else False
        onnx_name = args.names[i] if args.names and i < len(args.names) else f"model{i+1}.onnx"
        prec = "FP16" if use_half else "FP32"
        print(f"\n  [{i+1}] {run_name} -> {onnx_name} (imgsz={imgsz}, {prec})")

        model = YOLO(str(weights))
        model.export(format="onnx", imgsz=imgsz, half=use_half, opset=17, simplify=True)
        onnx_path = weights.with_suffix(".onnx")
        shutil.copy2(onnx_path, staging / onnx_name)

        size_mb = (staging / onnx_name).stat().st_size / 1024 / 1024
        total_onnx_mb += size_mb
        print(f"     ONNX: {size_mb:.1f} MB")

    # Copy ensemble run.py
    src_run = WORKDIR / "ensemble_run.py"
    shutil.copy2(src_run, staging / "run.py")
    print(f"\n  run.py copied")

    # Zip
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.iterdir()):
            zf.write(f, f.name)

    # Verify
    print(f"\nContents:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:30s} {info.file_size / 1024 / 1024:8.1f} MB")

    total = zip_path.stat().st_size / 1024 / 1024
    status = "OK" if total_onnx_mb <= MAX_SIZE_MB else "OVER WEIGHT LIMIT!"
    print(f"\n  Weights total: {total_onnx_mb:.1f} MB (limit: {MAX_SIZE_MB} MB) {status}")
    print(f"  Zip size: {total:.1f} MB")
    print(f"  Output: {zip_path}")
    print(f"\n  scp olivia:{zip_path} ~/Desktop/submission.zip")


if __name__ == "__main__":
    main()
