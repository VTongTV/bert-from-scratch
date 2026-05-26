from data.glue import GlueExample, GlueTask


class STSBTask(GlueTask):
    def __init__(self):
        super().__init__("STS-B", 1)

    def get_examples(self, lines):
        examples = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                score = float(parts[-1])
                examples.append(GlueExample(parts[5], parts[6], score))
        return examples
