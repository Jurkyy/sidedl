class Container:

    def __init__(self, settings):
        self.settings = settings
        self.components = []

    def add_component(self, component):
        self.components.append(component)
