import curses
from typing import Any, Callable, override


class ScrollableBox:
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    true_width: int
    true_height: int

    scrollable_x: bool
    scrollable_y: bool

    scroll_x: int
    scroll_y: int

    pad_obj: curses.window

    view_width: int
    view_height: int

    def __getattr__(self, name: str):
        if name == "view_width":
            return self.x_end - self.x_start
        elif name == "view_height":
            return self.y_end - self.y_start
        else:
            raise AttributeError

    def __init__(
        self,
        y: int,
        x: int,
        height: int,
        width: int,
        scrollable_y: bool = False,
        scrollable_x: bool = True,
    ):
        self.x_start = x
        self.y_start = y
        self.x_end = x + width
        self.y_end = y + height
        self.true_height = height
        self.true_width = width

        self.scrollable_x = scrollable_x
        self.scrollable_y = scrollable_y

        self.scroll_x = 0
        self.scroll_y = 0
        self.pad_obj = curses.newpad(height, width)

    def refresh(self, do_update: bool = True):
        if do_update:
            self.pad_obj.refresh(
                self.scroll_y,
                self.scroll_x,
                self.y_start,
                self.x_start,
                self.y_end,
                self.x_end,
            )
        else:
            self.pad_obj.noutrefresh(
                self.scroll_y,
                self.scroll_x,
                self.y_start,
                self.x_start,
                self.y_end,
                self.x_end,
            )

    def resize_true(self, height: int | None = None, width: int | None = None):
        """Updates the height and width of the underlying pad. This does not change how
        much of the screen the pad takes. Either argument can be specified or omitted.
        When omitted or set to None, that value is not changed.

        Args:
            height (int | None, optional): The height of the pad. Defaults to None,
                which keeps it the same.
            width (int | None, optional): The width of the pad. Defaults to None,
                which keeps it the same.
        """
        if height is None:
            height = self.true_height
        if width is None:
            width = self.true_width
        self.pad_obj.resize(height, width)
        self.true_height = height
        self.true_width = width

    def resize_view(self, height: int | None = None, width: int | None = None):
        """Updates the height and width of the drawn portion of the box. This does not change the
        size of the underlying pad. Either argument can be specified or omitted.
        When omitted or set to None, that value is not changed.

        Args:
            height (int | None, optional): The height of the pad. Defaults to None,
                which keeps it the same.
            width (int | None, optional): The width of the pad. Defaults to None,
                which keeps it the same.
        """
        if height is None:
            height = self.view_height
        if width is None:
            width = self.view_width
        self.y_end = self.y_start + height
        self.x_end = self.x_start + width

    def move(self, y: int, x: int, height: int | None = None, width: int | None = None):
        """Moves the drawn portion of the box to the given location.
        This does not change the size of the underlying
        pad. Takes additional arguments for resize_view()

        Args:
            y (int): y-position to move the upper-left corner to
            x (int): x-position to move the upper-left corner to
            height (int | None, optional): The height of the pad. Defaults to None,
                which keeps it the same.
            width (int | None, optional): The width of the pad. Defaults to None,
                which keeps it the same.
        """
        # we need to grab height and width now, since they are __getattr__'ed from x/y_start
        if height is None:
            height = self.view_height
        if width is None:
            width = self.view_width
        self.y_start = y
        self.x_start = x
        self.resize_view(height=height, width=width)


class ScrollableList(ScrollableBox):
    class ListEntry:
        increment: int
        draw_call: Callable[[curses.window, int, bool], None]
        id_str: str | None

        def __init__(
            self, increment: int, draw_call: Callable[[curses.window, int, bool], None]
        ):
            """Initializes a list entry

            Args:
                increment (int): Size of the entry in the scroll direction
                draw_call (Callable): callback used when drawn. Takes a
                    curses.window "parent" for which window to draw on,
                    an int "position" for the
                    scroll-direction position and boolean "is_hovered" for whether or not the
                    entry should be highlighted
            """
            self.increment = increment
            self.draw_call = draw_call
            self.id_str = None

    entries: list[ListEntry]
    entry_names: dict[str, int]
    selected_ind: int
    is_vertical_scroll: bool

    def __init__(
        self, y: int, x: int, height: int, width: int, vertical_scroll: bool = True
    ):
        super().__init__(
            y,
            x,
            height,
            width,
            scrollable_y=vertical_scroll,
            scrollable_x=not vertical_scroll,
        )
        self.selected_ind = 0
        self.entries = []
        self.entry_names = dict()
        self.is_vertical_scroll = vertical_scroll

    def __getattr__(self, name: str) -> Any:
        if name == "selected":
            return (
                self.entries[self.selected_ind].id_str
                if self.selected_ind < len(self.entries)
                else None
            )
        else:
            return super().__getattr__(name)

    @override
    def refresh(self, do_update: bool = True):
        self.pad_obj.erase()
        inc = 0
        for i, entry in enumerate(self.entries):
            entry.draw_call(self.pad_obj, inc, self.selected_ind == i)
            inc += entry.increment

        super().refresh(do_update=do_update)

    def add_entry(self, id_str: str, entry: ListEntry, replace: bool = True):
        # entries = [entry0, ...] specifies order/stores entry
        # (entry_names: name |-> index) makes so parent container does not need to know order.
        if id_str in self.entry_names:
            if not replace:
                raise ValueError(
                    f"Attempting to add entry {id_str} when it already exists, and the `replace` flag is not set."
                )
            self.entries[self.entry_names[id_str]] = entry
            entry.id_str = id_str
        else:
            self.entry_names[id_str] = len(self.entries)
            self.entries += [entry]
            entry.id_str = id_str

        # resize pad to ensure it is big enough
        # increase by one for padding, since writing to last character throws an error
        needed_size = 1
        for entry in self.entries:
            needed_size += entry.increment

        if self.is_vertical_scroll:
            if self.true_height < needed_size:
                self.resize_true(height=needed_size)
        else:
            if self.true_width < needed_size:
                self.resize_true(width=needed_size)

    def remove_entry(self, id_str: str, graceful=True):
        if id_str not in self.entry_names:
            if not graceful:
                raise ValueError(f"Attempting to remove nonexistent entry {id_str}")
        else:
            ind = self.entry_names[id_str]
            del self.entries[ind]
            del self.entry_names[id_str]

            # selected entry will have to change if an entry above it was removed
            # if selected entry was removed, select one above it (unless we are at the first one)
            if self.selected_ind >= ind and self.selected_ind > 0:
                self.selected_ind -= 1

            # same with all other indices, but we already removed the ind==0 case
            for st, ind_ in self.entry_names.items():
                if ind_ >= ind:
                    self.entry_names[st] -= 1

    def selection_up(self, amount: int = 1, scroll_if_needed: bool = True):
        self.selected_ind -= amount
        if self.selected_ind < 0:
            self.selected_ind = 0
        if scroll_if_needed:
            # ensure scroll_x/y is small enough to see the start of the entry
            entry_start = 0
            for i in range(self.selected_ind):
                entry_start += self.entries[i].increment

            if self.is_vertical_scroll:
                if self.scroll_y > entry_start:
                    self.scroll_y = entry_start
            else:
                if self.scroll_x > entry_start:
                    self.scroll_x = entry_start

    def selection_down(self, amount: int = 1, scroll_if_needed: bool = True):
        self.selected_ind += amount
        if self.selected_ind >= len(self.entries):
            self.selected_ind = len(self.entries) - 1
        if scroll_if_needed:
            # ensure scroll_x/y is large enough to see the end of the entry
            entry_end = -1
            for i in range(self.selected_ind + 1):
                entry_end += self.entries[i].increment

            if self.is_vertical_scroll:
                if self.scroll_y + self.view_height < entry_end:
                    self.scroll_y = entry_end - self.view_height
            else:
                if self.scroll_x + self.view_width < entry_end:
                    self.scroll_x = entry_end - self.view_width

    def manage_inputs(self, ch: int):
        if ch == (curses.KEY_UP if self.is_vertical_scroll else curses.KEY_LEFT):
            self.selection_up()
        if ch == (curses.KEY_DOWN if self.is_vertical_scroll else curses.KEY_RIGHT):
            self.selection_down()
