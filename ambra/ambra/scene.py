class Object:
    def __init__(self, name: str):
        self.name = name
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

class Scene:
    def __init__(self, name):
        self.name = name
        self.objects = []