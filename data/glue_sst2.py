from data.glue import GlueExample, GlueTask


class SST2Task(GlueTask):
    def __init__(self):
        super().__init__("SST-2", 2)

    def get_examples(self, lines):
        examples = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                label = int(parts[-1])
                examples.append(GlueExample(parts[0], "", label))
        return examples
