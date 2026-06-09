import argparse
import os
import time
import torch

from config.bert_config import BertConfig
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
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config = BertConfig(L=args.L, H=args.H, A=args.A, V=args.V, max_len=args.max_len)
    trainer = PreTrainer(config, device=args.device)
    if args.resume_from:
        trainer.load(args.resume_from)
        print(f"resumed from step {trainer.global_step}")
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"BERT pre-training: L={args.L} H={args.H} A={args.A} V={args.V}")
    print(f"Steps: {args.total_steps}, Warmup: {args.warmup_steps}, LR: {args.lr}")
    scaler = torch.amp.GradScaler("cuda") if args.fp16 else None
    start_time = time.time()
    while trainer.global_step < args.total_steps:
        batch = {
            "input_ids": torch.randint(0, config.V, (args.batch_size, args.max_len)),
            "segment_ids": torch.zeros(args.batch_size, args.max_len, dtype=torch.long),
            "attention_mask": torch.ones(args.batch_size, args.max_len),
            "mlm_labels": torch.full((args.batch_size, args.max_len), -100, dtype=torch.long),
            "nsp_labels": torch.randint(0, 2, (args.batch_size,)),
        }
        for i in range(args.batch_size):
            mask_pos = torch.randint(0, args.max_len, (int(args.max_len * 0.15),))
            batch["mlm_labels"][i, mask_pos] = torch.randint(0, config.V, (len(mask_pos),))
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = trainer.train_step(batch)
            scaler.scale(loss).backward()
            scaler.step(trainer.optimizer)
            scaler.update()
        else:
            loss = trainer.train_step(batch)
        if trainer.global_step % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"step {trainer.global_step} | loss {loss:.4f} | lr {trainer.scheduler.get_lr()[0]:.2e} | {elapsed:.1f}s")
        if trainer.global_step % args.save_every == 0 and trainer.global_step > 0:
            path = os.path.join(args.save_dir, f"ckpt-{trainer.global_step}.pt")
            trainer.save(path)
            print(f"saved checkpoint to {path}")


if __name__ == "__main__":
    main()
