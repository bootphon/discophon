import argparse
from pathlib import Path

from .pipeline import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phoneme Discovery benchmark")
    parser.add_argument("submission", type=Path, help="Path to the submission metadata JSON file")
    parser.add_argument("annotation", type=Path, help="Path to the annotation metadata JSON file")
    parser.add_argument("output", type=Path, help="Path to the output JSON file")
    parser.add_argument("--split", type=str, choices=["full", "dev", "test"])
    args = parser.parse_args()
    run_benchmark(args.submission, args.annotation, args.output, split=args.split)
