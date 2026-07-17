"""Offline evaluation of the trained posture models on the recorded dataset.

Runs entirely headless (no webcam, no mediapipe) — it loads the CSV landmark
dataset, rebuilds the same windows the trainer used, and reports overall +
per-class accuracy and a confusion matrix for each exercise's checkpoint.

IMPORTANT: train.py fits on ALL windows (no held-out split), so these numbers
are IN-SAMPLE (training-set) accuracy. They verify that the shipped weights,
feature pipeline, and label mapping are consistent end-to-end — not how the
model generalises to unseen recordings. See README "Model card".

Usage:
    python offline_eval.py                 # evaluate every exercise
    python offline_eval.py --mode Squat    # one exercise
    python offline_eval.py --mode Squat --limit 64   # quick CI smoke
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import FEEDBACK_MAP
from dataset import PostureDataset
from model import PostureModel

WEIGHTS_DIR = "weights"


def evaluate(exercise: str, limit: int | None = None, batch_size: int = 64):
    ckpt = os.path.join(WEIGHTS_DIR, f"{exercise}_model.pth")
    if not os.path.exists(ckpt):
        print(f"[skip] {exercise}: no checkpoint at {ckpt}")
        return None

    dataset = PostureDataset(exercise, augment=False)
    if len(dataset) == 0:
        print(f"[skip] {exercise}: no data")
        return None

    if limit is not None and limit < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(limit))

    classes = FEEDBACK_MAP[exercise]
    num_classes = len(classes)

    model = PostureModel(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    all_preds, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = float((preds == labels).mean())
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        conf[t][p] += 1

    print(f"\n===== {exercise} — {len(labels)} windows, {num_classes} classes =====")
    print(f"Overall (in-sample) accuracy: {acc * 100:.2f}%")
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        class_acc = float((preds[mask] == c).mean())
        print(f"  [{c}] {classes[c]:<32} {class_acc * 100:6.2f}%  (n={int(mask.sum())})")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(conf)
    return acc


def main():
    parser = argparse.ArgumentParser(description="Headless evaluation of trained posture models")
    parser.add_argument("--mode", choices=list(FEEDBACK_MAP.keys()), default=None,
                        help="Evaluate one exercise (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate at most N windows (smoke-test mode)")
    args = parser.parse_args()

    exercises = [args.mode] if args.mode else list(FEEDBACK_MAP.keys())
    results = {}
    for ex in exercises:
        acc = evaluate(ex, limit=args.limit)
        if acc is not None:
            results[ex] = acc

    if len(results) > 1:
        print("\n===== Summary (in-sample) =====")
        for ex, acc in results.items():
            print(f"  {ex:<14} {acc * 100:6.2f}%")

    if not results:
        raise SystemExit("No exercise could be evaluated (missing weights or data).")


if __name__ == "__main__":
    main()
