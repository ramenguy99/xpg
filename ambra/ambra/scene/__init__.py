from pyxpg import Context
from transform import Transform

class Object:
    def __init__(self, name: str):
        self.name = name
        self.transform = Transform.identity()
        self.children = None
    
    def create(ctx: Context):
        pass

    def render(ctx: Context):
        pass

    def destroy(ctx: Context):
        pass

class Scene(Object):
    def __init__(self, name):
        super().__init__(name)