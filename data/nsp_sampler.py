import random


class NSPSampler:
    def __init__(self, corpus):
        self.corpus = corpus

    def sample(self, rng=None):
        if rng is None:
            rng = random.Random()
        doc_idx = rng.randint(0, len(self.corpus) - 1)
        doc = self.corpus[doc_idx]
        if len(doc) < 2:
            return doc[0] if doc else "", "", True
        sent_idx = rng.randint(0, len(doc) - 2)
        text_a = doc[sent_idx]
        is_next = rng.random() < 0.5
        if is_next:
            text_b = doc[sent_idx + 1]
        else:
            text_b = self.corpus.get_random_sentence(rng)
        return text_a, text_b, is_next
