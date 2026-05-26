from data.glue import GlueExample, GlueTask


class MRPCCTask(GlueTask):
    def __init__(self):
        super().__init__("MRPC", 2)

    def get_examples(self, lines):
        examples = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                label = int(parts[0])
                examples.append(GlueExample(parts[3], parts[4], label))
        return examples
