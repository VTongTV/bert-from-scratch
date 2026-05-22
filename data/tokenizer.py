from data.tokens import CLS, SEP, UNK
from data.vocabulary import Vocabulary


class WordPieceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def tokenize(self, text):
        text = text.lower().strip()
        if not text:
            return []
        tokens = []
        for word in text.split():
            sub_tokens = self._wordpiece(word)
            tokens.extend(sub_tokens)
        return tokens

    def _wordpiece(self, word):
        if word in self.vocab:
            return [word]
        return self._greedy_longest_match(word)

    def _greedy_longest_match(self, word):
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            matched = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    tokens.append(substr)
                    matched = True
                    break
                end -= 1
            if not matched:
                tokens.append(UNK)
                start += 1
            else:
                start = end
        return tokens

    def _segment_subwords(self, text):
        words = text.strip().split()
        all_subwords = []
        for word in words:
            subwords = self._wordpiece(word)
            for i, sw in enumerate(subwords):
                if i > 0 and not sw.startswith("##") and sw != UNK:
                    subwords[i] = "##" + sw
            all_subwords.extend(subwords)
        return all_subwords

    def encode(self, text, max_len=512):
        tokens = self.tokenize(text)
        tokens = [CLS] + tokens[:max_len - 2] + [SEP]
        ids = [self.vocab.get_id(t) for t in tokens]
        return ids

    def encode_pair(self, text_a, text_b, max_len=512):
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b)
        max_tokens = max_len - 3
        while len(tokens_a) + len(tokens_b) > max_tokens:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        tokens = [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        ids = [self.vocab.get_id(t) for t in tokens]
        return ids, segment_ids

    def decode(self, ids):
        tokens = [self.vocab.get_token(i) for i in ids]
        text = ""
        for tok in tokens:
            if tok.startswith("##"):
                text += tok[2:]
            elif tok in (CLS, SEP):
                continue
            else:
                if text:
                    text += " "
                text += tok
        return text
