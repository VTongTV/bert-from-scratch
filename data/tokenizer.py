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
