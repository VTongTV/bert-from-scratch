from data.tokens import PAD, UNK, SPECIAL_TOKENS


class Vocabulary:
    def __init__(self, tokens=None):
        self.token_to_id = {}
        self.id_to_token = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok
        if tokens:
            for tok in tokens:
                if tok not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[tok] = idx
                    self.id_to_token[idx] = tok

    def __len__(self):
        return len(self.token_to_id)

    def __contains__(self, token):
        return token in self.token_to_id

    def get_id(self, token):
        return self.token_to_id.get(token, self.token_to_id[UNK])

    def get_token(self, idx):
        return self.id_to_token.get(idx, UNK)
