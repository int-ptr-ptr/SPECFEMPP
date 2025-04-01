import inspect
from typing import Any, Callable


def pass_supported_kwargs(
    func: Callable, kwargs: dict[str, Any] | None = None, **metakwargs
):
    if kwargs is None:
        kwargs = dict()
    kwargs.update(metakwargs)

    supported_kwargs = dict()
    pass_all = False
    for name, param in inspect.signature(func).parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            pass_all = True

        if name in kwargs:
            supported_kwargs[name] = kwargs[name]

    if pass_all:
        return func(**kwargs)
    else:
        return func(**supported_kwargs)
