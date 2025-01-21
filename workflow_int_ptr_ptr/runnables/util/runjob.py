import time
from multiprocessing import Process, Queue
import subprocess
import collections
from typing import Callable


class RunJob:
    def __init__(
        self,
        name: str,
        cmd: str | None = None,
        func: Callable | None = None,
        min_update_interval: int = 2,
        linebuf_size: int = 10,
        print_updates: bool = False,
    ):
        self.name = name
        self.cmd = cmd
        self.min_update_interval = min_update_interval
        self.linebuf_size = linebuf_size
        self.print_updates = print_updates
        self.is_complete = False
        self.func = func
        if (func is None) == (cmd is None):
            raise ValueError("Either `func` or `cmd` must be specified, and not both.")


def _run(job: RunJob, queue: Queue):
    tstart = time.time()
    tlast = tstart

    def log(st):
        m, s = divmod(int(round(time.time() - tstart)), 50)
        queue.put(f"[{job.name}: {m:4d}:{s:02d}] " + st)

    if job.cmd is None:
        try:
            # XOR on initialization, so we assume func is not None. (if it is, its a func fail)
            job.func(log)  # type: ignore
        except Exception:
            return False
        return True
    with subprocess.Popen(
        job.cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=True,
        bufsize=1,
        universal_newlines=True,
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


jobs = dict()
_queue = Queue()


def queue_job(job: RunJob):
    q = Queue()
    p = Process(target=_run, args=(job, q))
    p.start()
    i = 0
    while i in jobs:
        i += 1
    jobs[i] = (job, p, q)

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
