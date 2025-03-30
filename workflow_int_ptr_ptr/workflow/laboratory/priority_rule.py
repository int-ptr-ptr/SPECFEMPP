from enum import StrEnum, auto
from typing import override

from workflow.util.task_manager import Task


class RuleType(StrEnum):
    SET = auto()
    DEPENDENCY_CASCADE = auto()

    class EmptyRule:
        def __init__(self, config_entry=None):
            pass

        def apply(self, tasklist: list[Task]):
            pass

    class SetRule:
        taskID: str
        priority: float

        def __init__(self, config_entry):
            self.taskID = config_entry["task"]
            self.priority = float(config_entry["priority"])

        def apply(self, tasklist: list[Task]):
            for task in tasklist:
                if task.name == self.taskID:
                    task.priority = self.priority
                    break

    class DependencyCascadeRule(SetRule):
        def cascade(self, task: Task):
            task.priority = min(task.priority, self.priority)
            for dep in task.dependencies:
                self.cascade(dep)

        @override
        def apply(self, tasklist: list[Task]):
            for task in tasklist:
                if task.name == self.taskID:
                    self.cascade(task)
                    break

    def process_rule(self, config_entry):
        if self == RuleType.SET:
            return RuleType.SetRule(config_entry)
        elif self == RuleType.DEPENDENCY_CASCADE:
            return RuleType.DependencyCascadeRule(config_entry)

        return RuleType.EmptyRule(config_entry)

    @staticmethod
    def get_rule(config_entry):
        return RuleType(config_entry["rule"]).process_rule(config_entry)


Rule = RuleType.SetRule | RuleType.DependencyCascadeRule | RuleType.EmptyRule


class PriorityRule:
    rule: Rule

    def __init__(self, config_entry):
        self.rule = RuleType.get_rule(config_entry)

    def apply(self, tasklist: list[Task]):
        self.rule.apply(tasklist)
