from typing import Callable, Iterable, Sequence, override
import re
import time

from util.runjob import (
    RunJob,
    queue_job,
    consume_queue as consume_job_queue,
    is_job_running,
    complete_job as get_job_exitcode,
)
import util.curse_monitor


def _msg_strip_name(msg: str, keep_timestamp: bool = True) -> str:
    if m := re.match(r"\[(.+):\s*(\d+:\d\d)\](\s?)", msg):
        msg = msg[m.end() :]
        if keep_timestamp:
            msg = f"[{m.group(2)}]{m.group(3)}{msg}"
    return msg


class GroupContainer(util.curse_monitor.TestContainer):
    def __init__(self, groupname: str, numtasks: int | None = None):
        self.groupname = groupname
        super().__init__(f"Group: {groupname}")

        self.group_messages = GroupContainer.Task(groupname, numtasks=numtasks)
        self.tasks.append(self.group_messages)

    @override
    class Task(util.curse_monitor.TestContainer.Task):
        def __init__(self, groupname, *args, numtasks: int | None = None, **kwargs):
            if numtasks is None:
                super().__init__(f'Group "{groupname}"')
            else:
                super().__init__(f'Group "{groupname}": {numtasks} tasks')


class TaskContainer(util.curse_monitor.TestContainer):
    def __init__(
        self,
        task: "Task",
        group_container: GroupContainer | None = None,
    ):
        super().__init__(f"[{task.group}] {task.name}")
        self.tasks.append(self.Task(task.name, messages=list()))
        if group_container is not None:
            self.tasks.append(group_container.group_messages)

    def job_message_callback(self, msg: str):
        self.tasks[0].messages.append(msg)  # type: ignore


class Task:
    # the name of this task
    name: str

    # the corresponding job to run
    job: RunJob

    # group that this task belongs to
    group: str

    # list of tasks that must complete before this task can run
    dependencies: Sequence["Task"]

    # Callback, called in the main thread when the job completes.
    on_completion: Callable

    def __init__(
        self,
        job: RunJob,
        name: str = "unnamed job",
        group: str = "default",
        dependencies: list["Task"] | None = None,
        on_completion: Callable[[int], None] | None = None,
    ):
        self.name = name
        self.job = job
        self.group = group
        self.dependencies = list() if dependencies is None else dependencies
        self.on_completion = (
            (lambda x: None) if on_completion is None else on_completion
        )

    def generate_monitor_container(
        self, group_container: GroupContainer
    ) -> TaskContainer:
        disp = TaskContainer(self, group_container=group_container)
        return disp


class Manager:
    use_gui: bool
    running: bool

    # groups whose members must be done one at a time (ex: GPU intensive tasks)
    sequential_groups: list[str]
    tasks: list[Task]
    groups: dict[str, list[Task]]

    def __init__(
        self,
        use_gui: bool = False,
        tasks: list[Task] | None = None,
        sequential_groups: list[str] | None = None,
    ):
        self.use_gui = use_gui
        self.running = False
        self.sequential_groups = (
            list() if sequential_groups is None else sequential_groups
        )
        self.tasks = list()
        self.groups = dict()

        if tasks is not None:
            for t in tasks:
                self.add_task(t)

    def add_task(self, task: Task):
        self.tasks.append(task)
        grp = task.group
        if grp not in self.groups:
            self.groups[grp] = list()
        self.groups[grp].append(task)

    def add_tasks(self, *tasks: Iterable[Task | Iterable[Task]]):
        for task in tasks:
            if isinstance(task, Task):
                self.add_task(task)
            try:
                for subtask in task:
                    assert isinstance(subtask, Task)
                    self.add_task(subtask)
            except TypeError | AssertionError as e:
                raise TypeError(
                    "add_task() accepts only Tasks or Iterable[Task] arguments"
                ) from e

    def run(self):
        self.running = True
        with util.curse_monitor.TestMonitor(
            dummy_gui=not self.use_gui, close_with_key=False
        ) as gui:
            group_containers = dict()
            for group, tasks in self.groups.items():
                disp = GroupContainer(group)
                group_containers[group] = disp
            # integer list for deps for easy access
            queued_or_running: dict[int, list[int]] = {
                tindex: [
                    self.tasks.index(t) for t in task.dependencies if t in self.tasks
                ]
                for tindex, task in enumerate(self.tasks)
            }
            active_tasks: dict[int, tuple[int, TaskContainer]] = dict()

            def start_task(ID: int):
                task = self.tasks[ID]
                disp = task.generate_monitor_container(group_containers[task.group])
                active_tasks[ID] = (queue_job(task.job), disp)
                gui.add_tab(disp)

            dep_backlink: list[list[int]] = [list() for _ in range(len(self.tasks))]

            # initialize nondependent tasks and store indices of tasks that depend on these
            for t, deps in queued_or_running.items():
                for dep in deps:
                    # t depends on dep, so let dep have a reference to t for when it completes
                    dep_backlink[dep].append(t)
                if len(deps) == 0:
                    start_task(t)

            while queued_or_running:
                if not active_tasks:
                    raise Exception(
                        "Some tasks were not able to start. Did something fail,"
                        " or was there a circular dependency?"
                    )
                # consume active task queues
                for taskid in list(active_tasks.keys()):
                    jobid, container = active_tasks[taskid]
                    for line in consume_job_queue(jobid):
                        if gui.dummy_gui:
                            print(line, end="" if line.endswith("\n") else "\n")
                        else:
                            container.job_message_callback(_msg_strip_name(line))

                    if not is_job_running(jobid, false_on_nonempty_queue=True):
                        exitcode = get_job_exitcode(jobid)
                        self.tasks[taskid].on_completion(exitcode)
                        del queued_or_running[taskid]
                        gui.remove_tab(active_tasks[taskid][1])
                        del active_tasks[taskid]
                        if exitcode != 0:
                            continue
                        for dep in dep_backlink[taskid]:
                            queued_or_running[dep].remove(taskid)
                            if len(queued_or_running[dep]) == 0:
                                start_task(dep)

                gui.manage_inputs()
                gui.redraw_display()
                time.sleep(0.1)
