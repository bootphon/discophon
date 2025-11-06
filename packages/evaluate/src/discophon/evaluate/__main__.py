from pathlib import Path

from .evaluate import discovery_evaluation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("units", type=Path)
    parser.add_argument("phones", type=Path)
    parser.add_argument("--n-units", type=int, default=256)
    parser.add_argument("--n-phones", type=int, default=40)
    parser.add_argument("--step-units", type=int, default=20)
    parser.add_argument("--step-phones", type=int, default=10)
    args = parser.parse_args()
    print(
        discovery_evaluation(
            args.units,
            args.phones,
            n_units=args.n_units,
            n_phones=args.n_phones,
            step_units=args.step_units,
            step_phones=args.step_phones,
        )
    )
