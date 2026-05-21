from dataclasses import dataclass


@dataclass
class PretrainConfig:
    batch_size: int = 256
    max_steps: int = 1000000
    lr: float = 1e-4
    b1: float = 0.9
    b2: float = 0.999
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    seq_len_short: int = 128
    seq_len_long: int = 512
    short_seq_ratio: float = 0.9
    mlm_prob: float = 0.15
