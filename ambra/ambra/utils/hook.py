from typing import Any, Callable


class hook:  # noqa: N801
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self.fn = fn

    def __set_name__(self, owner: Any, name: str) -> None:
        func = self.fn

        def _decorator(self: Any, *args: Any, **kwargs: Any) -> Any:
            super_obj = super(owner, self)
            super_fn = getattr(super_obj, func.__name__)
            super_fn(*args, **kwargs)
            return func(self, *args, **kwargs)

        setattr(owner, name, _decorator)

    def __call__(self) -> None:
        raise AssertionError(
            "@hook_after decorator object should never be called directly. This can happen if you apply this decorator to a function that is not a method."
        )


class hook_before:  # noqa: N801
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self.fn = fn

    def __set_name__(self, owner: Any, name: str) -> None:
        func = self.fn

        def _decorator(self: Any, *args: Any, **kwargs: Any) -> Any:
            super_obj = super(owner, self)
            super_fn = getattr(super_obj, func.__name__)
            result = func(self, *args, **kwargs)
            super_fn(*args, **kwargs)
            return result

        setattr(owner, name, _decorator)

    def __call__(self) -> None:
        raise AssertionError(
            "@hook decorator object should never be called directly. This can happen if you apply this decorator to a function that is not a method."
        )
