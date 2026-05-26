from dataclasses import dataclass

from data.features import InputFeatures
from data.preprocessing import convert_examples_to_features


@dataclass
class GlueExample:
    text_a: str
    text_b: str = ""
    label: int = 0


class GlueTask:
    def __init__(self, name, num_labels):
        self.name = name
        self.num_labels = num_labels

    def get_examples(self, lines):
        raise NotImplementedError

    def get_features(self, tokenizer, examples, max_len=128):
        features = []
        for ex in examples:
            f = convert_examples_to_features(tokenizer, ex.text_a, ex.text_b, ex.label, max_len)
            features.append(f)
        return features
