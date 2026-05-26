from data.glue import GlueExample, GlueTask


class CoLATask(GlueTask):
    def __init__(self):
        super().__init__("CoLA", 2)

    def get_examples(self, lines):
        examples = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                label = int(parts[1])
                examples.append(GlueExample(parts[3], "", label))
        return examples
