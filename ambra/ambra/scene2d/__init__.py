from ..scene import Object, Scene
from ..transform2d import Transform

class Object2D(Object):
    def __init__(self, name):
        super().__init__(name)

        self.transform = Transform.identity()

class Scene2D(Scene):
    pass