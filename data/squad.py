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
