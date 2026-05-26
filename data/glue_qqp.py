from data.glue import GlueExample, GlueTask


class QQPTask(GlueTask):
    def __init__(self):
        super().__init__("QQP", 2)

    def get_examples(self, lines):
        examples = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                label = int(parts[-1])
                examples.append(GlueExample(parts[3], parts[4], label))
        return examples
