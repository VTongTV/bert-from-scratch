import random

from data.tokens import MASK


def mask_token(token, rng, vocab_size, mask_prob=0.8):
    if rng.random() < mask_prob:
        return MASK
    return token


def random_replace(token, rng, vocab_size):
    return rng.randint(0, vocab_size - 1)


def keep_unchanged(token, rng, vocab_size):
    return token


def apply_mlm_mask(input_ids, rng, vocab_size, mlm_prob=0.15):
    masked_ids = list(input_ids)
    labels = [-1] * len(input_ids)
    for i in range(len(input_ids)):
        if rng.random() < mlm_prob:
            labels[i] = input_ids[i]
            r = rng.random()
            if r < 0.8:
                masked_ids[i] = vocab_size + 1
            elif r < 0.9:
                masked_ids[i] = rng.randint(0, vocab_size - 1)
    return masked_ids, labels
