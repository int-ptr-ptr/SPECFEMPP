from typing import Callable

from util.runjob import FunctionJob


class SeismoPlotJob(FunctionJob):
    def __init__(
        self,
        name: str,
        min_update_interval: int = 0,
        linebuf_size: int = 10,
    ):
        super().__init__(
            name,
            self.plot,
            min_update_interval=min_update_interval,
            linebuf_size=linebuf_size,
            print_updates=True,
        )

    def plot(self, logfunc: Callable):
        pass
