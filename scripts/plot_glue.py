import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="glue_results.png")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.results) as f:
        data = json.load(f)
    tasks = list(data.keys())
    scores = list(data.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(tasks, scores)
    ax.set_ylabel("Score")
    ax.set_title("GLUE Task Results")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
