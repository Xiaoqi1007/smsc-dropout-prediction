"""Train the fixed four-source model, evaluate, and verify saved predictions."""

from pathlib import Path
import argparse, sys

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "verify"])
    parser.add_argument("--config", type=Path, default=P / "configs/analysis.json")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    from smsc.config import load_config

    spec = load_config(args.config)
    run = P / "results/experiments" / spec["name"]
    if args.command == "train":
        from smsc.training import train

        print(train(args.config, P, args.seed, args.threads))
    else:
        from smsc.evaluation import evaluate_run, verify_run

        print((evaluate_run if args.command == "evaluate" else verify_run)(run, P))


if __name__ == "__main__":
    main()
