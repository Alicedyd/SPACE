#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Summarize accuracies from results.csv files across subfolders (each subfolder = one ckpt).
# Usage:
#   python summarize_ckpts.py /path/to/root_dir
#   # optional arguments:
#   python summarize_ckpts.py /path/to/root_dir --acc-col accuracy
#   python summarize_ckpts.py /path/to/root_dir --pattern results.csv --recursive
#   python summarize_ckpts.py /path/to/root_dir --save /path/to/summary.csv
#
# What it does:
# - Traverses the immediate subdirectories (or recursively with --recursive).
# - For each subdir, looks for a CSV file (default name: results.csv).
# - Reads the CSV and computes an average accuracy with robust auto-detection:
#     1) Preferred columns (case-insensitive):
#        ["accuracy","acc","top1","top1_acc","val_accuracy","val_acc","test_accuracy"]
#     2) If not found but both ["correct","total"] exist => uses sum(correct) / sum(total) (percent)
#     3) If --acc-col is provided, uses that column exactly (case-insensitive match)
# - Produces a summary table and prints the ckpt (subdir name) with the highest accuracy.
# - Optionally saves the summary table to CSV via --save.

import argparse
import sys
from pathlib import Path
import pandas as pd

PREFERRED_ACC_COLS = [
    "accuracy", "acc", "top1", "top1_acc", "val_accuracy", "val_acc", "test_accuracy"
]

def find_results_csv(d: Path, pattern: str = "results.csv"):
    """Return the results csv path inside directory d matching the pattern.
    If multiple match, prefer exact filename first, otherwise pick the first in sorted order.
    """
    exact = d / pattern
    if exact.exists():
        return exact
    # fallback: glob for pattern
    matches = sorted(d.glob(pattern))
    if matches:
        return matches[0]
    return None

def autodetect_accuracy(df: pd.DataFrame, forced_col: str | None = None) -> float | None:
    cols_lower = {c.lower(): c for c in df.columns}
    # -- forced column path
    if forced_col:
        key = forced_col.lower()
        if key in cols_lower:
            series = pd.to_numeric(df[cols_lower[key]], errors="coerce").dropna()
            if len(series) == 0:
                return None
            return float(series.mean())
        return None

    # -- preferred columns
    for name in PREFERRED_ACC_COLS:
        if name in cols_lower:
            series = pd.to_numeric(df[cols_lower[name]], errors="coerce").dropna()
            if len(series) == 0:
                continue
            return float(series.mean())

    # -- correct/total path (weighted)
    if "correct" in cols_lower and "total" in cols_lower:
        correct = pd.to_numeric(df[cols_lower["correct"]], errors="coerce").fillna(0)
        total = pd.to_numeric(df[cols_lower["total"]], errors="coerce").fillna(0)
        denom = total.sum()
        if denom > 0:
            return float((correct.sum() / denom) * 100.0)
        return None

    # nothing found
    return None

def summarize(root_dir: Path, pattern: str, recursive: bool, acc_col: str | None):
    rows = []
    it = root_dir.rglob("*") if recursive else root_dir.glob("*")
    for p in it:
        if p.is_dir():
            csv_path = find_results_csv(p, pattern=pattern)
            if not csv_path or not csv_path.is_file():
                continue
            try:
                df = pd.read_csv(csv_path, engine="python")
            except Exception as e:
                rows.append({"ckpt": p.name, "csv": str(csv_path), "mean_accuracy": None, "error": f"read_csv: {e}"})
                continue

            acc = autodetect_accuracy(df, forced_col=acc_col)
            if acc is None:
                rows.append({"ckpt": p.name, "csv": str(csv_path), "mean_accuracy": None, "error": "No accuracy column detected"})
            else:
                rows.append({"ckpt": p.name, "csv": str(csv_path), "mean_accuracy": float(acc), "error": ""})

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Summarize accuracies from results.csv across subfolders.")
    parser.add_argument("root", type=str, help="Root directory containing ckpt subfolders")
    parser.add_argument("--pattern", type=str, default="results.csv", help="CSV filename or glob pattern to search for (default: results.csv)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into nested subfolders")
    parser.add_argument("--acc-col", type=str, default=None, help="Explicit accuracy column name to use (case-insensitive)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the summary CSV")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Root directory not found or not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    df = summarize(root, pattern=args.pattern, recursive=args.recursive, acc_col=args.acc_col)
    if df.empty:
        print("[ERROR] No results found. Check your --pattern or directory structure.", file=sys.stderr)
        sys.exit(2)

    # report best
    df_valid = df.dropna(subset=["mean_accuracy"])
    if df_valid.empty:
        print("[ERROR] Found CSVs but could not detect an accuracy column in any of them.", file=sys.stderr)
        print(df.to_string(index=False))
        sys.exit(3)

    # Normalize accuracy scale: if values look like 0-1, convert to percentage
    if df_valid["mean_accuracy"].max() <= 1.0:
        df_valid["mean_accuracy"] = df_valid["mean_accuracy"] * 100.0
        # merge back
        df = df.merge(df_valid[["ckpt", "mean_accuracy"]], on="ckpt", how="left", suffixes=("", "_norm"))
        df["mean_accuracy"] = df["mean_accuracy_norm"].fillna(df["mean_accuracy"])
        df.drop(columns=["mean_accuracy_norm"], inplace=True)

    best_row = df_valid.loc[df_valid["mean_accuracy"].idxmax()]
    best_ckpt = best_row["ckpt"]
    best_acc = float(best_row["mean_accuracy"])

    print("\n=== Summary (sorted by accuracy desc) ===")
    print(df.sort_values(by=["mean_accuracy"], ascending=False).to_string(index=False))

    print(f"\n>>> Best ckpt: {best_ckpt}  (mean_accuracy = {best_acc:.4f})")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(by=["mean_accuracy"], ascending=False).to_csv(out_path, index=False)
        print(f"[Saved] Summary CSV -> {out_path}")

if __name__ == "__main__":
    main()
