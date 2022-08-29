import flaml.automl.task.factory
from .automl import AutoML as AutoMLGeneric

# Hackity hack hack
# The below is ultra-grim, but allows us to break the TS_FORECAST
# logic into its own class which maintaining backwards compatibility of
# the public interface to AutoML
class AutoML(AutoMLGeneric):
    def __init__(self, **settings):
        super().__init__(**settings)

    def fit(self, *args, **kwargs):
        # Is it in kwargs?
        task_name = kwargs.get("task")
        # Is it in args?
        if len(args) >= 6:
            task_name = args[5]
        # Is it in self._settings?
        task_name = task_name or self._settings["task"]
        self.task = flaml.automl.task.factory.task_factory(task_name)

        super().fit(*args, **kwargs)


# Set the docstring for sphinx autodocs
AutoML.__init__.__doc__ = AutoMLGeneric.__init__.__doc__
AutoML.fit.__doc__ = AutoMLGeneric.fit.__doc__
