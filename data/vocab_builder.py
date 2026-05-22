from data.tokens import SPECIAL_TOKENS
from data.vocabulary import Vocabulary


def build_vocab_from_corpus(texts, min_count=1, max_vocab=30000):
    counts = {}
    for text in texts:
        for word in text.strip().split():
            counts[word] = counts.get(word, 0) + 1
    tokens = sorted(
        (t for t, c in counts.items() if c >= min_count),
        key=lambda t: (-counts[t], t),
    )
    tokens = tokens[:max_vocab - len(SPECIAL_TOKENS)]
    subword_tokens = set()
    for tok in tokens:
        subword_tokens.add(tok)
        subword_tokens.add("##" + tok)
    return Vocabulary(list(subword_tokens))
