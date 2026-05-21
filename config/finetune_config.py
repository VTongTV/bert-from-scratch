from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    batch_size: int = 32
    lr: float = 5e-5
    epochs: int = 3
    P_drop: float = 0.1
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
