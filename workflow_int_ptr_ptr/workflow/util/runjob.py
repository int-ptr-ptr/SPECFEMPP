import collections
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue, queues
from typing import Callable


class RunJob:
    def __init__(
        self,
        name: str,
        min_update_interval: int = 2,
        linebuf_size: int = 10,
        print_updates: bool = False,
    ):
        self.name = name
        self.min_update_interval = min_update_interval
        self.linebuf_size = linebuf_size
        self.print_updates = print_updates
        self.is_complete = False
        self.was_submitted = False


class FunctionJob(RunJob):
    func: Callable

    def __init__(
        self,
        name: str,
        func: Callable[[Callable], None | bool],
        min_update_interval: int = 2,
        linebuf_size: int = 10,
        print_updates: bool = False,
    ):
        super().__init__(
            name=name,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=print_updates,
        )
        self.func = func


class CommunicationQueuedFunctionJob(FunctionJob):
    def __init__(
        self,
        name: str,
        func: Callable[[Callable, Queue, Queue], None | bool],
        min_update_interval: int = 2,
        linebuf_size: int = 10,
        print_updates: bool = False,
    ):
        super().__init__(
            name=name,
            func=lambda x: None,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=print_updates,
        )
        self.func = func


class SystemCommandJob(RunJob):
    def __init__(
        self,
        name: str,
        cmd: str,
        min_update_interval: int = 2,
        linebuf_size: int = 10,
        print_updates: bool = False,
        cwd: str | None = None,
    ):
        super().__init__(
            name=name,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=print_updates,
        )
        self.cmd = cmd
        self.cwd = cwd


def _run(job: RunJob, queue: Queue, **kwargs) -> None:
    tstart = time.time()
    tlast = tstart

    def log(st):
        m, s = divmod(int(round(time.time() - tstart)), 50)
        queue.put(f"[{job.name}: {m:4d}:{s:02d}] " + st)

    if isinstance(job, FunctionJob):
        try:
            if isinstance(job, CommunicationQueuedFunctionJob):
                ret = job.func(log, kwargs["comm_to_job"], kwargs["comm_from_job"])
            else:
                ret = job.func(log)

            if (ret is None) or (isinstance(ret, bool) and ret):
                # function is successful
                return
            else:
                sys.exit(ret)
        except Exception as e:
            raise e
    elif isinstance(job, SystemCommandJob):
        with subprocess.Popen(
            job.cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True,
            bufsize=1,
            universal_newlines=True,
            cwd=job.cwd,
        ) as popen:
            output_queue = collections.deque(maxlen=job.linebuf_size)

            # capture outputs until the end of the program
            if popen.stdout is not None:
                for line in popen.stdout:
                    output_queue.append(line)
                    t = time.time()
                    if t - tlast > job.min_update_interval:
                        if job.print_updates:
                            log(line)
                        tlast = t

            if popen.stderr is not None:
                for line in popen.stderr:
                    output_queue.append(line)
            retcode = popen.wait()

        if retcode != 0:
            log(f"subprocess ({job.name}) failed! Output:\n" + "".join(output_queue))
            sys.exit(retcode)
        elif job.print_updates:
            log(f"subprocess ({job.name}) completed!")
            return


@dataclass
class JobStorage:
    job: RunJob
    process: Process
    logqueue: Queue
    comm_to_job: queues.Queue | None = None
    comm_from_job: queues.Queue | None = None
    unconsumed_log_entries: list | None = None
    _all_log_entries_consumed: bool = False
    retcode: int = -1
    _is_alive: bool = True

    def handle_completion(self):
        """Checks if the process completed, and updates the storage state accordingly."""
        if self.process.is_alive() or not self._is_alive:
            # either still alive, or we already handled the completion
            return

        self._is_alive = False

        # consume_queue() will call handle_completion again, but we set _is_alive already.
        self.unconsumed_log_entries = self.consume_queue()
        self.retcode = -1 if self.process.exitcode is None else self.process.exitcode
        self.process.join(0.01)

    def is_alive(self):
        self.handle_completion()
        return self._is_alive

    def consume_queue(self) -> list:
        self.handle_completion()
        if self.unconsumed_log_entries is not None:
            self._all_log_entries_consumed = True
            return self.unconsumed_log_entries

        q = self.logqueue
        L = list()
        while not q.empty():
            L.append(q.get())
        return L


jobs: dict[int, JobStorage] = dict()


def queue_job(job: RunJob):
    if job.was_submitted:
        raise ValueError("Cannot queue an already submitted job!")

    job.was_submitted = True
    q = Queue()

    # new jobID
    i = 0
    while i in jobs:
        i += 1

    if isinstance(job, CommunicationQueuedFunctionJob):
        tojob = Queue()
        fromjob = Queue()
        p = Process(
            target=_run,
            args=(job, q),
            kwargs={"comm_to_job": tojob, "comm_from_job": fromjob},
        )
        jobs[i] = JobStorage(job, p, q, tojob, fromjob)
    else:
        p = Process(target=_run, args=(job, q))
        jobs[i] = JobStorage(job, p, q)
    p.start()

    return i

    # while procs:
    #     time.sleep(1)
    #     while not queue.empty():
    #         print(queue.get())
    #     for name, p in list(procs.items()):
    #         if not p.is_alive():
    #             print(f"[!] process for {name} has completed!")
    #             p.join(0.01)
    #             del procs[name]


def consume_queue(jid: int) -> list:
    if jid in jobs:
        return jobs[jid].consume_queue()
    return []


def is_job_running(jid: int, false_on_nonempty_queue: bool = True):
    if jid not in jobs:
        return False

    if false_on_nonempty_queue and jobs[jid]._all_log_entries_consumed:
        return False

    return jobs[jid].is_alive()


def complete_job(jid: int) -> int | None:
    """Garbage-collects the job and retrieves an exit code if done.

    Args:
        jid (int): The job ID.

    Returns:
        int | None: the exit code, or None if the job is still running or already consumed.
    """
    if jid not in jobs:
        return None

    if is_job_running(jid):
        return None

    retcode = jobs[jid].retcode
    del jobs[jid]

    return retcode


def get_job_queues(jid: int) -> dict[str, Queue]:
    if jid in jobs:
        job_storage = jobs[jid]
        queues: dict[str, Queue] = {"log": job_storage.logqueue}
        if job_storage.comm_from_job is not None:
            queues["from_job"] = job_storage.comm_from_job
        if job_storage.comm_to_job is not None:
            queues["to_job"] = job_storage.comm_to_job

        return queues
    return dict()
