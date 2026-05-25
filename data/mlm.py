import random

from data.tokens import MASK


def mask_token(token, rng, vocab_size, mask_prob=0.8):
    if rng.random() < mask_prob:
        return MASK
    return token


def random_replace(token, rng, vocab_size):
    return rng.randint(0, vocab_size - 1)
