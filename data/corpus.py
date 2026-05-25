class DocumentCorpus:
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]

    def get_random_sentence(self, rng, doc_idx=None):
        if doc_idx is None:
            doc_idx = rng.randint(0, len(self.documents))
        doc = self.documents[doc_idx]
        if not doc:
            return ""
        sent_idx = rng.randint(0, len(doc) - 1)
        return doc[sent_idx]
