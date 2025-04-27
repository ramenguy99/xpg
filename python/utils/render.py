from pyxpg import *
from typing import TypeVar


R = TypeVar("R")

class PerFrameResource:
    def __init__(self, Resource: R, n: int, *args, **kwargs):
        self.resources = []
        for _ in range(n):
            self.resources.append(Resource(*args, **kwargs))
        self.index = 0
    
    def get_current(self) -> R:
        return self.resources[self.index]
    
    def advance(self):
        self.index = (self.index + 1) % len(self.resources)
    
    def get_current_and_advance(self) -> R:
        cur = self.get_current()
        self.advance()
        return cur