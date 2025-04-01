import os
import time
from dataclasses import dataclass

import workflow.util.config as config

gpulock = config.get("lock.gpu")
gpulocklock = gpulock + ".write"
lock_timeout = float(config.get("lock.timeout"))

_IO_attempt_delay = 1


class _write_lock:
    timestamp: float | None
    timeout: float

    def __init__(self, timeout: float = -1):
        self.timestamp = None
        self.timeout = timeout

    def _acquire_lock(self) -> bool:
        try:
            with open(gpulocklock, "x") as f:
                timestamp = time.time()
                f.write(f"{os.getpid()},{timestamp}")
                self.timestamp = timestamp
            return True
        except FileExistsError:
            try:
                with open(gpulocklock, "r") as f:
                    pid, t = f.read().split(",")
                    if int(pid) == os.getpid():
                        # somehow, this PID has the lock.
                        self.timestamp = float(t)
                        return True
                    if time.time() - float(t) > lock_timeout:
                        # lock has timed out
                        os.remove(gpulocklock)
            except Exception:
                ...
            return False

    def still_has_lock(self):
        return (
            self.timestamp is not None and time.time() - self.timestamp < lock_timeout
        )

    def __enter__(self):
        t0 = time.time()
        while not self._acquire_lock():
            if self.timeout >= 0 and time.time() - t0 > self.timeout:
                raise TimeoutError
            time.sleep(_IO_attempt_delay)

        return None

    def __exit__(self, esc_type, exc_val, exc_tb):
        self.timestamp = None
        if os.path.exists(gpulocklock):
            os.remove(gpulocklock)


class GPULock:
    @dataclass
    class Receipt:
        is_serial: bool
        is_gpu: bool

    @dataclass
    class LockEntry:
        uid: int
        pid: int
        expiration: float
        utilization: float

    utilization: float
    alloted_time: float
    timestamp: float | None
    uid: int | None
    request_timeout: float
    enter_serial_context_on_fail: float

    def __init__(
        self,
        utilization: float = 1.0,
        alloted_time: float = 3600,
        request_timeout: float = -1,
        enter_serial_context_on_fail: bool = False,
    ):
        self.utilization = utilization
        self.request_timeout = request_timeout
        self.alloted_time = alloted_time
        self.timestamp = None
        self.uid = None
        self.enter_serial_context_on_fail = enter_serial_context_on_fail

    @staticmethod
    def _readlockfile(
        ignorePID: int | None = None, ignoreUID: int | None = None
    ) -> tuple[list[LockEntry], float]:
        entries = []
        utilization = 0
        try:
            with open(gpulock, "r") as f:
                for line in f:
                    uid, pid, t, u = line.split(",")
                    uid = int(uid)
                    pid = int(pid)
                    t = float(t)
                    u = float(u)
                    if (
                        (ignoreUID is None or uid != ignoreUID)
                        and (ignorePID is None or pid != ignorePID)
                        and time.time() < t
                    ):
                        entries.append(
                            GPULock.LockEntry(
                                uid=uid, pid=pid, expiration=t, utilization=u
                            )
                        )
                        utilization += u
        except IOError:
            ...
        return entries, utilization

    @staticmethod
    def _writelockfile(entries: list[LockEntry]):
        with open(gpulock, "w") as f:
            for entry in entries:
                f.write(
                    f"{entry.uid},{entry.pid},{entry.expiration},{entry.utilization}\n"
                )

    def _clear_self(self):
        if self.uid is not None:
            with _write_lock():
                GPULock._writelockfile(GPULock._readlockfile(ignoreUID=self.uid)[0])

    def _try_to_add_self(self) -> bool:
        if self.uid is not None:
            return False
        with _write_lock():
            entries, utilization = GPULock._readlockfile()
            if utilization + self.utilization <= 1.0 + 1e-8:
                self.timestamp = time.time()
                used_ids = {lock.uid for lock in entries}
                self.uid = 0
                while self.uid in used_ids:
                    self.uid += 1
                entries.append(
                    GPULock.LockEntry(
                        uid=self.uid,
                        pid=os.getpid(),
                        expiration=self.timestamp + self.alloted_time,
                        utilization=self.utilization,
                    )
                )
                GPULock._writelockfile(entries)
                return True
        return False

    def __enter__(self):
        t0 = time.time()
        while not self._try_to_add_self():
            if self.request_timeout >= 0 and time.time() - t0 > self.request_timeout:
                self.timestamp = None
                self.uid = None
                return GPULock.Receipt(is_serial=True, is_gpu=False)
            time.sleep(_IO_attempt_delay)

        return GPULock.Receipt(is_serial=False, is_gpu=True)

    def __exit__(self, esc_type, exc_val, exc_tb):
        self._clear_self()
        self.timestamp = None
        self.uid = None


# print(GPULock._readlockfile())
# with GPULock(utilization=0.1):
#     with GPULock(utilization=0.2):
#         print(GPULock._readlockfile())
#     print(GPULock._readlockfile())
# print(GPULock._readlockfile())
