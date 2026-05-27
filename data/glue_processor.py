from data.glue_mnli import MNLITask
from data.glue_qqp import QQPTask
from data.glue_qnli import QNLITask
from data.glue_sst2 import SST2Task
from data.glue_cola import CoLATask
from data.glue_stsb import STSBTask
from data.glue_mrpc import MRPCCTask
from data.glue_rte import RTETask
from data.glue_wnli import WNLITask


GLUE_TASKS = {
    "MNLI": MNLITask,
    "QQP": QQPTask,
    "QNLI": QNLITask,
    "SST-2": SST2Task,
    "CoLA": CoLATask,
    "STS-B": STSBTask,
    "MRPC": MRPCCTask,
    "RTE": RTETask,
    "WNLI": WNLITask,
}


def get_glue_task(name):
    return GLUE_TASKS[name]()


def load_glue_data(task_name, split, data_dir):
    task = get_glue_task(task_name)
    filepath = f"{data_dir}/{task_name}/{split}.tsv"
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return task.get_examples(lines)
