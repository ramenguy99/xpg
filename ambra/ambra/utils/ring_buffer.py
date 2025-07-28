from typing import TypeVar, Type, List, Generic


R = TypeVar("R")

# TODO: likely modify this to just take in a list, not sure why we need the Type and args stuff
class RingBuffer(Generic[R]):
    def __init__(self, n: int, typ: Type, *args, **kwargs):
        self.items: List[R] = []
        for _ in range(n):
            self.items.append(typ(*args, **kwargs))
        self.index = 0

    def __len__(self):
        return self.items.__len__()

    def __iter__(self):
        return self.items.__iter__()

    def __next__(self):
        return self.items.__next__()

    def get_current(self) -> R:
        return self.items[self.index]

    def advance(self):
        self.index = (self.index + 1) % len(self.items)

    def set(self, index: int):
        self.index = index % len(self.items)

    def get_current_and_advance(self) -> R:
        cur = self.get_current()
        self.advance()
        return cur

    def __repr__(self):
        return f'[{", ".join([str(i) for i in self.items])}]'
