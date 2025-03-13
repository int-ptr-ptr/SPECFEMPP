import curses
import curses.panel
from collections import deque
from typing import Iterable

from .ui import ProgressBar, ScrollableList, UnscrollableMessageLog


class TestContainer:
    """Holds data for tabs in test_monitor.
    Has no TUI content, so no context manager is needed.
    """

    class Task:
        name: str
        progress: float
        messages: Iterable[str]

        def __init__(
            self, name: str, progress: float = 0, messages: Iterable[str] | None = None
        ):
            self.name = name
            self.progress = progress
            self.messages = list() if messages is None else messages

    name: str
    message: str
    progress: float
    tasks: list[Task]

    def __init__(self, name):
        self.name = name
        self.message = ""
        self.progress = 0
        self.tasks = list()


class TestMonitor:
    """Handles TUI for tests. TestMonitor is formatted with a left sidebar listing tests
    which can be highlighted for a more detailed view.
    """

    NAME_OFFSET_RIGHT = 3
    NAME_OFFSET_LEFT = 3
    PROGRESSBAR_OFFSET_RIGHT = 3

    active: bool
    tests: dict[str, TestContainer]
    _tab_windows: dict[str, curses.panel.panel]
    _sidebar_width: int  # if zero, set to curses.COLS // 4

    sidebar: ScrollableList
    sidebar_progress_template: ProgressBar
    sidebar_progressbar_offset_start: int
    hovering_sidebar: bool

    close_with_key: bool

    @staticmethod
    def default_sidebar_width() -> int:
        return curses.COLS // 4

    def sidebar_width(self):
        return (
            (
                TestMonitor.default_sidebar_width()
                if self._sidebar_width < 0
                else self._sidebar_width
            )
            if self.active
            else -1
        )

    def __init__(
        self,
        sidebar_width: int = -1,
        dummy_gui: bool = False,
        close_with_key: bool = True,
    ):
        self.tests = dict()
        self._tab_windows = dict()
        self.active = False
        self.stdscr = None
        self._sidebar_width = sidebar_width
        self.hovering_sidebar = True
        self.width = -1
        self.height = -1
        self.sidebar_progressbar_offset_start = -1
        self.dummy_gui = dummy_gui
        self.close_with_key = close_with_key

    def redraw_tab(self, name: str):
        if not self.active:
            return
        win = self._tab_windows[name].window()
        container = self.tests[name]
        win.erase()
        win.border()
        win.addstr(0, 5, f"[{container.name}]")

        numtasks = len(container.tasks)
        if numtasks > 0:
            my, mx = win.getmaxyx()
            taskwidth = (mx - 2) // numtasks
            logger = UnscrollableMessageLog(my - 3, taskwidth - 1)
            progress = ProgressBar(taskwidth - 3)
            for tnum, task in enumerate(container.tasks):
                xstart = taskwidth * tnum + 1
                # namestring
                progress_st = f" ({task.progress * 100:5.1f}% )"
                headerlen = taskwidth - 1 - len(progress_st)
                header = (
                    task.name + (" " * (headerlen - len(task.name)))
                    if len(task.name) < headerlen
                    else (task.name[: headerlen - 3] + "...")
                )

                win.addstr(1, xstart, header + progress_st)
                win.addstr(2, xstart, ProgressBar.START_CHAR)
                win.addstr(
                    progress.get_str(task.progress),
                    curses.color_pair(curses.COLOR_GREEN),
                )
                win.addstr(ProgressBar.END_CHAR)
                logger.draw_to(win, 3, xstart, task.messages)

                if tnum > 0:
                    win.addch(0, xstart - 1, curses.ACS_BSSS)
                    win.vline(1, xstart - 1, curses.ACS_SBSB, my - 2)
                    win.addch(my - 1, xstart - 1, curses.ACS_SSBS)

    def update_sidebar_progressbar(self):
        if not self.active:
            return
        # initialize progress bar
        name_pad = (
            max(len(test.name) for test in self.tests.values())
            if len(self.tests) > 0
            else 0
        )
        self.sidebar_progressbar_offset_start = (
            self.NAME_OFFSET_LEFT + name_pad + self.NAME_OFFSET_RIGHT
        )  # left side buffer
        sidebar_width = self.sidebar_width()
        progressbarsize = (
            sidebar_width
            - self.sidebar_progressbar_offset_start
            - self.PROGRESSBAR_OFFSET_RIGHT
            - 2  # first and last markers
        )
        self.sidebar_progress_template = ProgressBar(progressbarsize)

    def _load_tab(self, name: str):
        if not self.active:
            return
        if name not in self._tab_windows:
            # initialize curses.window

            sidebar = self.sidebar_width()
            win = curses.newwin(curses.LINES, curses.COLS - sidebar, 0, sidebar)
            self._tab_windows[name] = curses.panel.new_panel(win)
            self.redraw_tab(name)

            def draw(parent, inc, highlight, testind):
                testname = self.tests[testind].name

                sidebar_width = self.sidebar_width()
                parent.addstr(
                    inc,
                    self.sidebar_progressbar_offset_start,
                    ProgressBar.START_CHAR,
                    (curses.A_BOLD * highlight),
                )
                parent.addstr(
                    self.sidebar_progress_template.get_str(
                        self.tests[testind].progress
                    ),
                    (curses.A_BOLD * highlight) | curses.color_pair(curses.COLOR_GREEN),
                )
                parent.addstr(
                    ProgressBar.END_CHAR,
                    (curses.A_BOLD * highlight),
                )
                parent.addstr(
                    inc,
                    self.NAME_OFFSET_LEFT,
                    testname,
                    (curses.A_BOLD | (curses.A_UNDERLINE * self.hovering_sidebar))
                    * highlight,
                )
                parent.addstr(
                    inc + 1,
                    self.NAME_OFFSET_LEFT,
                    self.tests[testind].message,
                    curses.A_BOLD,
                )

                # mark start and end of progress bar, since unfilled has no indicator

                # markers for tab entries
                parent.addch(inc, 2, "▔")
                parent.addch(inc + 1, 2, "▁")
                parent.addch(inc, sidebar_width - 3, "▔")
                parent.addch(inc + 1, sidebar_width - 3, "▁")
                if highlight:
                    if self.hovering_sidebar:
                        parent.addstr(inc, 0, "▐██")
                        parent.addstr(inc + 1, 0, "▐██")
                        parent.addstr(inc, sidebar_width - 3, "██▌")
                        parent.addstr(inc + 1, sidebar_width - 3, "██▌")
                    else:
                        parent.addstr(inc, 0, "▐▔")
                        parent.addstr(inc + 1, 0, "▐▁")
                        parent.addstr(inc, sidebar_width - 2, "▔▌")
                        parent.addstr(inc + 1, sidebar_width - 2, "▁▌")

                else:
                    parent.addch(inc, 1, "▐")
                    parent.addch(inc + 1, 1, "▐")
                    parent.addch(inc, sidebar_width - 2, "▌")
                    parent.addch(inc + 1, sidebar_width - 2, "▌")

            self.sidebar.add_entry(
                name,
                ScrollableList.ListEntry(
                    2,
                    lambda parent, inc, highlight, ind=name: draw(
                        parent, inc, highlight, ind
                    ),
                ),
            )

    def _unload_tab(self, name: str):
        if name in self._tab_windows:
            del self._tab_windows[name]
        self.sidebar.remove_entry(name)

    def __enter__(self):
        if self.dummy_gui:
            return self
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)
        curses.start_color()
        curses.use_default_colors()
        for i in range(0, curses.COLORS):
            curses.init_pair(i, i, -1)
        self.active = True
        for name in self.tests.keys():
            self._load_tab(name)
        self.sidebar = ScrollableList(0, 0, curses.LINES - 1, self.sidebar_width())
        self.update_sidebar_progressbar()
        try:
            # hide cursor
            curses.curs_set(0)
            curses.setsyx(-1, -1)
        except Exception:
            ...

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.dummy_gui:
            return
        curses.nocbreak()
        if self.stdscr:
            self.stdscr.keypad(False)
            self.stdscr.nodelay(False)
        curses.start_color()
        curses.echo()
        curses.endwin()
        self.active = False
        self.stdscr = None

    def add_tab(self, tab: TestContainer):
        self.tests[tab.name] = tab
        if self.active:
            self._load_tab(tab.name)
            self.update_sidebar_progressbar()

    def remove_tab(self, tab: TestContainer | str):
        if isinstance(tab, TestContainer):
            tab = tab.name

        if self.active:
            self._unload_tab(tab)
            self.update_sidebar_progressbar()
        del self.tests[tab]

    def draw_sidebar(self, do_update: bool = True):
        if not self.active:
            return
        self.sidebar.refresh(do_update=do_update)

    def on_resize(self):
        if self.active and self.stdscr is not None:
            curses.update_lines_cols()
            sidebar = self.sidebar_width()
            self.sidebar.resize_view(height=curses.LINES - 1, width=sidebar)
            self.sidebar.resize_true(width=sidebar)
            self.update_sidebar_progressbar()
            for w in self._tab_windows.values():
                win = w.window()
                win.resize(curses.LINES - 1, curses.COLS - sidebar)
                win.mvwin(0, sidebar)
                # self.redraw_tab(name)

    def manage_inputs(self, error_out_on_close: bool = False):
        if not self.active:
            return
        if self.stdscr is not None:
            while (c := self.stdscr.getch()) != curses.ERR:
                # handle key c
                if c == curses.KEY_RESIZE:
                    self.on_resize()
                elif self.close_with_key and (c == ord("q") or c == ord("Q")):
                    self.active = False
                    if error_out_on_close:
                        raise StopIteration
                    break
                elif c == curses.KEY_RIGHT:
                    self.hovering_sidebar = False
                elif c == curses.KEY_LEFT:
                    self.hovering_sidebar = True
                if self.hovering_sidebar:
                    self.sidebar.manage_inputs(c)

    def redraw_display(self):
        if not self.active:
            return
        self.draw_sidebar(do_update=False)
        sel = self.sidebar.selected
        if sel is not None:
            self._tab_windows[sel].top()
            self.redraw_tab(sel)
        curses.panel.update_panels()
        curses.doupdate()


if __name__ == "__main__":
    import time

    with TestMonitor() as m:
        tests = [TestContainer(f"Test {i}") for i in range(20)]
        for tab in tests:
            m.add_tab(tab)
            tab.message = f"..... yes. ({tab.name})"

        def randstr(length: int):
            s = ""
            for i in range(length):
                s += chr(ord("a") + hash(time.time() + i) % 26)
            return s

        tests[0].tasks.append(TestContainer.Task("aaa", 0.3))
        tests[0].tasks.append(TestContainer.Task("bbb", 1))
        tests[1].tasks.append(TestContainer.Task("ccc", 0.3))
        tests[0].tasks[0].messages = [randstr(400) for _ in range(10)]
        tests[0].tasks[1].messages = deque()
        tests[1].tasks[0].messages = deque()

        tests[2].tasks.append(
            TestContainer.Task("a super long header: " + randstr(500), 0)
        )
        tests[2].tasks.append(TestContainer.Task("st2", 0.333))
        tests[2].tasks.append(TestContainer.Task("st3", 0.667))
        tests[2].tasks.append(TestContainer.Task("st4", 1))
        tests[2].tasks[0].messages = [randstr(600) for _ in range(10)]
        # m._tab_windows[0].border()
        # m._tab_windows[0].refresh()  # type: ignore
        t0 = time.time()

        tlast = t0
        while m.active:
            for tab in tests:
                tab.progress += (hash(tab.name) % 50) / 10000
                while tab.progress > 1:
                    tab.progress -= 1

            tests[0].tasks[1].progress += 4 / 1000
            while tests[0].tasks[1].progress > 1:
                tests[0].tasks[1].progress -= 1

            t = time.time()
            if t - tlast > 1:
                tlast = t
                tests[0].tasks[1].messages.append(f"Log [{int(t - t0)}]: {randstr(30)}")
                tests[0].tasks[1].messages.append(f"Log [{int(t - t0)}]: {randstr(30)}")
                tests[0].tasks[1].messages.append(f"Log [{int(t - t0)}]: {randstr(30)}")
                tests[1].tasks[0].messages.append(
                    f"Log [{int(t - t0)}]: {randstr(400)}"
                )
                while len(tests[0].tasks[1].messages) > 100:
                    tests[0].tasks[1].messages.popleft()
                while len(tests[1].tasks[0].messages) > 100:
                    tests[1].tasks[0].messages.popleft()
            m.manage_inputs()
            m.redraw_display()
            time.sleep(0.01)

        # for _ in range(6):
        #     m.sidebar.selected = (m.sidebar.selected + 1) % len(m.sidebar.entries)
        #     m.draw_sidebar()
        #     time.sleep(1)

        # for y in range(0, m.sidebar.true_height-1):
        #     for x in range(0, m.sidebar.true_width-1):
        #         m.sidebar.pad_obj.addch(y,x, ord('a') + (x*x+y*y) % 26)
        # m.sidebar.pad_obj.addstr(3,3, f"({m.sidebar.y_end},{m.sidebar.x_end}) -> {curses.LINES},{curses.COLS}")
        # m.draw_sidebar()
