import argparse
import os
import torch

from config.bert_config import BertConfig
from training.finetune_trainer import FinetuneTrainer


GLUE_TASKS = {
    "cola": 2, "sst-2": 2, "mrpc": 2, "sts-b": 1,
    "qqp": 2, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=list(GLUE_TASKS.keys()))
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--H", type=int, default=768)
    parser.add_argument("--A", type=int, default=12)
    parser.add_argument("--V", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="glue_output")
    return parser.parse_args()


def main():
    args = parse_args()
    num_classes = GLUE_TASKS[args.task]
    config = BertConfig(L=args.L, H=args.H, A=args.A, V=args.V, max_len=args.max_len)
    trainer = FinetuneTrainer(config, num_classes, device=args.device, lr=args.lr, epochs=args.epochs)
    print(f"GLUE task: {args.task}, classes: {num_classes}")
    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")


if __name__ == "__main__":
    main()
