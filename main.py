from __future__ import annotations

import argparse
from utils import run_pipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Digits SVC demo (refactored)")
    ap.add_argument("--gamma", type=float, default=0.001, help="SVC gamma")
    ap.add_argument("--test-size", type=float, default=0.5, help="test split size")
    ap.add_argument("--shuffle", action="store_true", help="shuffle train/test split")
    ap.add_argument("--no-plots", action="store_true", help="disable matplotlib windows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        gamma=args.gamma,
        test_size=args.test_size,
        shuffle=args.shuffle,
        show_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
