from dataclasses import dataclass


@dataclass
class InputFeatures:
    input_ids: list
    segment_ids: list
    attention_mask: list
    label: int = 0
