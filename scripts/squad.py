import argparse
import torch

from config.bert_config import BertConfig
from model.bert import BertModel
from model.qa_head import QAHead, qa_loss, best_span_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=12)
    parser.add_argument("--H", type=int, default=768)
    parser.add_argument("--A", type=int, default=12)
    parser.add_argument("--V", type=int, default=30000)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BertConfig(L=args.L, H=args.H, A=args.A, V=args.V, max_len=args.max_len)
    model = BertModel(config).to(args.device)
    head = QAHead(config.H).to(args.device)
    print(f"SQuAD fine-tuning: L={args.L} H={args.H} A={args.A}")


if __name__ == "__main__":
    main()
