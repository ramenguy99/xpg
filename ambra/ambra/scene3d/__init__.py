# from ..renderer import Renderer
from ..transform3d import Transform
from ..scene import Object, Scene

class Object3D(Object):
    def __init__(self, name: str):
        self.name = name
        self.transform = Transform.identity()
        self.children = []
    
    # def create(self, renderer: Renderer):
    def create(self, renderer):
        pass

    # def render(self, renderer: Renderer):
    def render(self, renderer, frame):
        pass

    # def destroy(self, renderer: Renderer):
    def destroy(self, renderer):
        pass

class Scene3D(Scene):
    pass
    