import json


class SquadReader:
    def read(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        examples = []
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    qas_id = qa["id"]
                    answers = []
                    for answer in qa["answers"]:
                        answers.append({
                            "text": answer["text"],
                            "start": answer["answer_start"],
                        })
                    examples.append({
                        "id": qas_id,
                        "question": question,
                        "context": context,
                        "answers": answers,
                    })
        return examples


def extract_spans(context, answer_text, answer_start, tokenizer):
    context_tokens = context.split()
    char_idx = 0
    token_start = None
    token_end = None
    for i, token in enumerate(context_tokens):
        if char_idx == answer_start:
            token_start = i
        char_idx += len(token) + 1
        if char_idx >= answer_start + len(answer_text) and token_end is None:
            token_end = i
    if token_start is None:
        token_start = 0
    if token_end is None:
        token_end = 0
    return token_start, token_end
