import json
from dataclasses import asdict

from config.bert_config import BertConfig


def save_config(config, path):
    with open(path, "w") as f:
        json.dump(asdict(config), f)


def load_config(path):
    with open(path, "r") as f:
        d = json.load(f)
    return BertConfig(**d)
