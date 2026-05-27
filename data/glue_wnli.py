from data.glue import GlueExample, GlueTask


class WNLITask(GlueTask):
    def __init__(self):
        super().__init__("WNLI", 2)

    def get_examples(self, lines):
        examples = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                label = int(parts[-1])
                examples.append(GlueExample(parts[1], parts[2], label))
        return examples
