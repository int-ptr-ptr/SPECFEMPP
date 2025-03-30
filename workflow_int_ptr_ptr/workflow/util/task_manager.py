import itertools
import queue
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, override

from workflow.util import curse_monitor
from workflow.util.runjob import (
    RunJob,
    is_job_running,
    queue_job,
)
from workflow.util.runjob import (
    complete_job as get_job_exitcode,
)
from workflow.util.runjob import (
    consume_queue as consume_job_queue,
)


def _msg_strip_name(msg: str, keep_timestamp: bool = True) -> str:
    if m := re.match(r"\[(.+):\s*(\d+:\d\d)\](\s?)", msg):
        msg = msg[m.end() :]
        if keep_timestamp:
            msg = f"[{m.group(2)}]{m.group(3)}{msg}"
    return msg


class GroupContainer(curse_monitor.TestContainer):
    def __init__(self, groupname: str, numtasks: int | None = None):
        self.groupname = groupname
        super().__init__(f"Group: {groupname}")

        self.group_messages = GroupContainer.Task(groupname, numtasks=numtasks)
        self.tasks.append(self.group_messages)

    @override
    class Task(curse_monitor.TestContainer.Task):
        def __init__(self, groupname, *args, numtasks: int | None = None, **kwargs):
            if numtasks is None:
                super().__init__(f'Group "{groupname}"')
            else:
                super().__init__(f'Group "{groupname}": {numtasks} tasks')


class TaskContainer(curse_monitor.TestContainer):
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

    # group(s) that this task belongs to
    group: list[str]

    # list of tasks that must complete before this task can run
    dependencies: Sequence["Task"]

    # Callback, called in the main thread when the job completes.
    on_completion: Callable

    # Callback, called in the main thread before the job starts
    on_pre_run: Callable

    # priority: lower number is selected first (0 is default)
    priority: float

    def __init__(
        self,
        job: RunJob,
        name: str = "unnamed job",
        group: list[str] | str | None = None,
        dependencies: list["Task"] | None = None,
        on_completion: Callable[[int], None] | None = None,
        on_pre_run: Callable[[], None] | None = None,
        priority: float = 0,
    ):
        self.name = name
        self.job = job
        if group is None:
            self.group = []
        elif isinstance(group, str):
            self.group = [group]
        else:
            self.group = group
        self.dependencies = list() if dependencies is None else dependencies
        self.on_completion = (
            (lambda x: None) if on_completion is None else on_completion
        )
        self.on_pre_run = (lambda: None) if on_pre_run is None else on_pre_run
        self.priority = priority

    def get_primary_group(self) -> str | None:
        return None if not self.group else self.group[0]

    def generate_monitor_container(
        self, group_container: GroupContainer
    ) -> TaskContainer:
        disp = TaskContainer(self, group_container=group_container)
        return disp


@dataclass(order=True)
class _PrioTask:
    priority: float = field(compare=True)
    taskID: int = field(compare=False)


class Manager:
    use_gui: bool
    running: bool

    # groups whose members must be done one at a time (ex: GPU intensive tasks)
    sequential_groups: list[str]

    # all tasks handled by this manager
    tasks: list[Task]

    # all tasks that belong to each group
    groups: dict[str, list[Task]]

    # all tasks by group[0]. This is a partition.
    primary_groups: dict[str | None, list[Task]]

    # print lots of logs? Logs are not printed when GUI is live
    verbose: bool

    # the total number of tasks active at once cannot exceed this (negative is unlimited)
    max_concurrent_tasks: int

    def __init__(
        self,
        use_gui: bool = False,
        tasks: list[Task] | None = None,
        sequential_groups: list[str] | None = None,
        verbose: bool = False,
        max_concurrent_tasks: int = -1,
    ):
        self.verbose = verbose
        self.use_gui = use_gui
        self.running = False
        self.sequential_groups = (
            list() if sequential_groups is None else sequential_groups
        )
        self.tasks = list()
        self.groups = dict()
        self.primary_groups = dict()
        self.max_concurrent_tasks = max_concurrent_tasks

        if tasks is not None:
            for t in tasks:
                self.add_task(t)

    def _log(self, *args, **kwargs):
        if self.verbose and not (self.use_gui and self.running):
            print(*args, **kwargs)

    def add_task(self, task: Task):
        self.tasks.append(task)
        grp = task.group
        self._log(f"Adding task {task.name} to manager (groups = {task.group}).")

        if task.get_primary_group() not in self.primary_groups:
            self.primary_groups[task.get_primary_group()] = list()
        self.primary_groups[task.get_primary_group()].append(task)

        for g in grp:
            if g not in self.groups:
                self.groups[g] = list()
            self.groups[g].append(task)

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
        if self.running:
            raise RuntimeError("Attempting to run() an already running task manager.")

        if not self.tasks:
            self._log("Manager.run() call issued without any tasks. Doing nothing.")
            return

        self.running = True
        with curse_monitor.TestMonitor(
            dummy_gui=not self.use_gui, close_with_key=False
        ) as gui:
            self._log(
                f"Manager.run() call issued with {len(self.tasks)} tasks."
                f" Entering TestMonitor context with dummy_gui={gui.dummy_gui}."
            )
            group_containers = dict()
            for group in self.primary_groups.keys():
                grpname = "(ungrouped)" if group is None else group
                disp = GroupContainer(grpname)
                group_containers[group] = disp
            # integer list for deps for easy access
            queued_or_running: dict[int, list[int]] = {
                tindex: [
                    self.tasks.index(t) for t in task.dependencies if t in self.tasks
                ]
                for tindex, task in enumerate(self.tasks)
            }

            sequential_task_waitlist: set[int] = set()
            # taskID -> (jobID, gui element)
            active_tasks: dict[int, tuple[int, TaskContainer]] = dict()
            active_sequential_groups: set[str] = set()

            active_queue: queue.PriorityQueue[_PrioTask] = queue.PriorityQueue()

            def start_task(ID: int, queued_removals: set[int] | None = None):
                task = self.tasks[ID]
                for group in task.group:
                    if (
                        group in self.sequential_groups
                        and group in active_sequential_groups
                    ):
                        if queued_removals is None:
                            self._log(f"{task.name} (id={ID}) [blocked], ", end="")
                            sequential_task_waitlist.add(ID)
                        return

                if queued_removals is None:
                    self._log(f"{task.name} (id={ID}), ", end="")
                else:
                    self._log(f"  Queued Task {task.name} (id={ID}) unblocked.")
                    queued_removals.add(ID)
                for group in task.group:
                    if group in self.sequential_groups:
                        active_sequential_groups.add(group)

                active_queue.put(_PrioTask(priority=task.priority, taskID=ID))

            dep_backlink: list[list[int]] = [list() for _ in range(len(self.tasks))]

            # initialize nondependent tasks and store indices of tasks that depend on these
            self._log("Initially enqueued tasks: [", end="")
            for t, deps in queued_or_running.items():
                for dep in deps:
                    # t depends on dep, so let dep have a reference to t for when it completes
                    dep_backlink[dep].append(t)
                if len(deps) == 0:
                    start_task(t)

            self._log("]")
            if (not active_tasks) and (not active_queue):
                raise RuntimeError(
                    "All tasks have a dependency. Cannot start due to deadlock."
                )

            while queued_or_running:
                if (not active_tasks) and (not active_queue):
                    raise Exception(
                        "Some tasks were not able to start. Did something fail,"
                        " or was there a circular dependency?"
                    )
                # pop the active queue to capacity
                while (
                    self.max_concurrent_tasks < 0
                    or len(active_tasks) < self.max_concurrent_tasks
                ) and (not active_queue.empty()):
                    ID = active_queue.get_nowait().taskID
                    task = self.tasks[ID]
                    task.on_pre_run()
                    disp = task.generate_monitor_container(
                        group_containers[task.get_primary_group()]
                    )
                    active_tasks[ID] = (queue_job(task.job), disp)
                    gui.add_tab(disp)

                # consume active task job queues
                for taskid in list(active_tasks.keys()):
                    jobid, container = active_tasks[taskid]
                    for line in consume_job_queue(jobid):
                        if gui.dummy_gui:
                            print(line, end="" if line.endswith("\n") else "\n")
                        container.job_message_callback(_msg_strip_name(line))

                    if not is_job_running(jobid, true_on_nonempty_queue=True):
                        exitcode = get_job_exitcode(jobid, error_on_still_running=True)
                        self._log(
                            f"Task {self.tasks[taskid].name} (id={taskid}) exited with code {exitcode}."
                        )
                        self.tasks[taskid].on_completion(exitcode)
                        del queued_or_running[taskid]
                        gui.remove_tab(active_tasks[taskid][1])
                        del active_tasks[taskid]

                        # refresh active sequential groups
                        active_sequential_groups.clear()
                        actives = itertools.chain(
                            active_tasks.keys(),
                            (entry.taskID for entry in active_queue.queue),
                        )
                        for active_tid in actives:
                            for group in self.tasks[active_tid].group:
                                active_sequential_groups.add(group)

                        if exitcode == 0:
                            self._log("  unlocked dependencies: [", end="")
                            # start in order of task priority
                            q: queue.PriorityQueue[_PrioTask] = queue.PriorityQueue()
                            for dep in dep_backlink[taskid]:
                                queued_or_running[dep].remove(taskid)
                                if len(queued_or_running[dep]) == 0:
                                    q.put_nowait(
                                        _PrioTask(
                                            priority=self.tasks[dep].priority,
                                            taskID=dep,
                                        )
                                    )
                            while not q.empty():
                                start_task(q.get_nowait().taskID)
                            self._log("]")

                        # check for sequential group unlock
                        seq_remove = set()

                        # start in order of task priority
                        q: queue.PriorityQueue[_PrioTask] = queue.PriorityQueue()
                        for tid in sequential_task_waitlist:
                            q.put(
                                _PrioTask(priority=self.tasks[tid].priority, taskID=tid)
                            )
                        while not q.empty():
                            start_task(
                                q.get_nowait().taskID, queued_removals=seq_remove
                            )
                        for tid in seq_remove:
                            sequential_task_waitlist.remove(tid)

                gui.manage_inputs()
                gui.redraw_display()
                time.sleep(0.1)
            self._log("Completed all tasks. Exiting TextMonitor context.")
        self.running = False
