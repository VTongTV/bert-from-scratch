from dataclasses import dataclass


@dataclass
class BertConfig:
    L: int = 12
    H: int = 768
    A: int = 12
    d_ff: int = 3072
    V: int = 30000
    max_len: int = 512
    P_drop: float = 0.1

    def __post_init__(self):
        self.d_ff = 4 * self.H


def bert_base_config():
    return BertConfig(L=12, H=768, A=12)
