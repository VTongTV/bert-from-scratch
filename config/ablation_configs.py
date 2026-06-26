from config.bert_config import BertConfig


def small_config():
    return BertConfig(L=3, H=768, A=12)


def medium_config_3():
    return BertConfig(L=6, H=768, A=3)


def medium_config_12():
    return BertConfig(L=6, H=768, A=12)


def large_h_config():
    return BertConfig(L=12, H=1024, A=16)
