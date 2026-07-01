import argparse
import json
import torch

from config.bert_config import BertConfig
from model.bert import BertModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="step_eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"evaluating checkpoints from {args.checkpoint_dir}")
    results = {"steps": [], "accuracy": []}
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"results saved to {args.output}")


if __name__ == "__main__":
    main()
