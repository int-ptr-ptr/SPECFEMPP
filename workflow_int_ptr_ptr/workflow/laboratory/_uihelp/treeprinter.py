import os

from workflow.util.task_manager import Task


def terminal_num_cols() -> int:
    return os.get_terminal_size().columns


class TaskTree:
    tasklist: list[Task]

    def __init__(self, tasks: list[Task] | None = None):
        self.tasklist = list() if tasks is None else tasks
