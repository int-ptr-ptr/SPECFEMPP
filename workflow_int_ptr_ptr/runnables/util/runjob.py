import time
from multiprocessing import Process, Queue
import subprocess
import collections
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


def _run(job: RunJob, queue: Queue, **kwargs) -> bool:
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
                return True
            else:
                return False
        except Exception:
            return False
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
            return False
        elif job.print_updates:
            log(f"subprocess ({job.name}) completed!")
        return True

    return False


jobs = dict()
_queue = Queue()


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
        jobs[i] = (job, p, q, tojob, fromjob)
    else:
        p = Process(target=_run, args=(job, q))
        jobs[i] = (job, p, q)
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


def consume_queue(jid: int):
    if jid in jobs:
        q = jobs[jid][2]
        if isinstance(q, type(_queue)):
            L = list()
            while not q.empty():
                L.append(q.get())
            return L

        L = jobs[jid][2]

        jobs[jid][1].join(0.01)
        del jobs[jid]
        return L

    return []


def is_job_running(jid: int):
    if jid not in jobs:
        return False

    if jobs[jid][1].is_alive():
        return True

    L = consume_queue(jid)
    if jid in jobs:
        jobs[jid] = (jobs[jid][0], jobs[jid][1], L)


def get_job_queues(jid: int) -> dict[str, Queue]:
    if jid in jobs:
        job_tuple = jobs[jid]
        if isinstance(job_tuple[0], FunctionJob):
            return {
                "log": job_tuple[2],
                "to_job": job_tuple[3],
                "from_job": job_tuple[4],
            }
        else:
            return {
                "log": job_tuple[2],
            }
    return dict()
