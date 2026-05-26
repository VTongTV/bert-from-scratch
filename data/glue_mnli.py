from data.glue import GlueExample, GlueTask


class MNLITask(GlueTask):
    def __init__(self):
        super().__init__("MNLI", 3)

    def get_examples(self, lines):
        examples = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 10:
                label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
                label = label_map.get(parts[-1], 0)
                examples.append(GlueExample(parts[8], parts[9], label))
        return examples
