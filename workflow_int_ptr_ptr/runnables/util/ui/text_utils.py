import curses
from typing import Iterable
from collections import deque

PROGRESS_CHARS = " ▎▎▍▌▋▊▉█"
PROGRESS_FULLCHAR = PROGRESS_CHARS[-1]
PROGRESS_SUBDIVS = len(PROGRESS_CHARS) - 1  # equate fullchar and emptychar


class ProgressBar:
    START_CHAR = "▕"
    END_CHAR = "▏"

    string_length: int
    _ticks: int

    def __init__(self, string_length: int):
        self.string_length = string_length
        # |xxxxxx|   (-2 from first and last characters)
        self._ticks = self.string_length * PROGRESS_SUBDIVS

    def get_str(self, ratio: float):
        ratio = min(1, max(0, ratio))
        progress_fill = int(round(self._ticks * ratio))
        if progress_fill == self._ticks:
            return f"{PROGRESS_FULLCHAR * self.string_length}"
        progress_subinc = progress_fill % PROGRESS_SUBDIVS
        progress_bars_full = progress_fill // PROGRESS_SUBDIVS

        return (
            (PROGRESS_FULLCHAR * progress_bars_full)
            + PROGRESS_CHARS[progress_subinc]
            + (" " * (self.string_length - progress_bars_full - 1))
        )


class UnscrollableMessageLog:
    width: int
    height: int

    def __init__(self, height: int, width: int):
        self.width = width
        self.height = height

    def draw_to(self, window: curses.window, y: int, x: int, messages: Iterable[str]):
        outs = deque()
        for message in messages:
            for line in message.split("\n"):
                # wrapping
                while line:
                    outs.append(line[: self.width])
                    line = line[self.width :]
                    if len(outs) >= self.height:
                        outs.popleft()
        while outs:
            window.addstr(y, x, outs.popleft())
            y += 1
