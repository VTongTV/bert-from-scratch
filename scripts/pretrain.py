import argparse
import torch

from config.bert_config import BertConfig, bert_base_config
from config.pretrain_config import PretrainConfig
from training.pretrain_trainer import PreTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--H", type=int, default=768)
    parser.add_argument("--A", type=int, default=12)
    parser.add_argument("--V", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--total_steps", type=int, default=1000000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BertConfig(L=args.L, H=args.H, A=args.A, V=args.V, max_len=args.max_len)
    trainer = PreTrainer(config, device=args.device)
    print(f"BERT pre-training: L={args.L} H={args.H} A={args.A} V={args.V}")
    print(f"Steps: {args.total_steps}, Warmup: {args.warmup_steps}, LR: {args.lr}")


if __name__ == "__main__":
    main()
