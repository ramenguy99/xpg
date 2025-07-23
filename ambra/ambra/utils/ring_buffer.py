from typing import TypeVar, Type


R = TypeVar("R")

class RingBuffer:
    def __init__(self, n: int, typ: Type, *args, **kwargs):
        self.items = []
        for _ in range(n):
            self.items.append(typ(*args, **kwargs))
        self.index = 0
    
    def get_current(self) -> R:
        return self.items[self.index]
    
    def advance(self):
        self.index = (self.index + 1) % len(self.items)
    
    def get_current_and_advance(self) -> R:
        cur = self.get_current()
        self.advance()
        return cur
