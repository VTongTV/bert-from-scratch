import csv


class SwagReader:
    def read(self, filepath):
        examples = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                examples.append({
                    "id": row["video-id"],
                    "sent1": row["sent1"],
                    "sent2": row["sent2"],
                    "ending0": row["ending0"],
                    "ending1": row["ending1"],
                    "ending2": row["ending2"],
                    "ending3": row["ending3"],
                    "label": int(row.get("label", 0)),
                })
        return examples
