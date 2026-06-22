import re
import string
import collections


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score_qa(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_common = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    if num_common == 0:
        return 0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_evaluate(preds, golds):
    em_total = 0
    f1_total = 0
    for pred, gold in zip(preds, golds):
        em_total += exact_match(pred, gold)
        f1_total += f1_score_qa(pred, gold)
    n = len(preds)
    return {"em": em_total / n * 100, "f1": f1_total / n * 100}
