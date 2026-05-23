from data.tokens import CLS, SEP, PAD
from data.tokenizer import WordPieceTokenizer
from data.features import InputFeatures


def convert_examples_to_features(tokenizer, text_a, text_b=None, label=0, max_len=512):
    if text_b is None:
        ids, segment_ids = tokenizer.encode_pair(text_a, "", max_len)
    else:
        ids, segment_ids = tokenizer.encode_pair(text_a, text_b, max_len)
    attention_mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(tokenizer.vocab.get_id(PAD))
        segment_ids.append(0)
        attention_mask.append(0)
    return InputFeatures(
        input_ids=ids[:max_len],
        segment_ids=segment_ids[:max_len],
        attention_mask=attention_mask[:max_len],
        label=label,
    )
