from dataclasses import dataclass


@dataclass
class PretrainExample:
    input_ids: list
    segment_ids: list
    attention_mask: list
    masked_ids: list
    mlm_labels: list
    nsp_label: int
