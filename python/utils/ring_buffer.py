from typing import Generic, Iterator, List, TypeVar

R = TypeVar("R")


class RingBuffer(Generic[R]):
    def __init__(self, items: List[R]):
        self.items = items
        self.index = 0

    def __len__(self) -> int:
        return self.items.__len__()

    def __iter__(self) -> Iterator[R]:
        return self.items.__iter__()

    def __repr__(self) -> str:
        return self.items.__repr__()

    def get_current(self) -> R:
        return self.items[self.index]

    def advance(self) -> None:
        self.index = (self.index + 1) % len(self.items)

    def set(self, index: int) -> None:
        self.index = index % len(self.items)

    def get_current_and_advance(self) -> R:
        cur = self.get_current()
        self.advance()
        return cur
