import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default="model_size_ablation.png")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.results) as f:
        data = json.load(f)
    names = list(data.keys())
    params = [data[n]["params"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, params)
    ax.set_ylabel("Parameters")
    ax.set_title("Model Size Ablation")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
