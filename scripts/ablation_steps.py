import argparse
import torch

from config.bert_config import BertConfig, bert_base_config
from training.pretrain_trainer import PreTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    config = bert_base_config()
    print(f"training step ablation: {args.steps} steps")


if __name__ == "__main__":
    main()
