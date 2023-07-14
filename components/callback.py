from components import Component
from packages.deeplearning.callbacks.callback_caller import *
from components.decorators.database_decorators import callback_database


class CallbackComponent(Component):

    def __init__(self, settings):
        super().__init__(settings)
        self.callback_functions = {}

    @callback_database
    def add_callback(self, callback_type, callback_name=None, **kwargs):
        callback_function = callback_functions(callback_type)(**kwargs)
        if callback_name is None:
            self.callback_functions[callback_function.__name__] = callback_function(**kwargs)
            return callback_function.__name__
        else:
            self.callback_functions[callback_name] = callback_function(**kwargs)
            return callback_name
