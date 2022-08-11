from . import tasks
from .automl import AutoML as AutoMLGeneric
from .time_series import AutoMLTS
from ..data import TS_FORECASTREGRESSION


# Hackity hack hack
# The below is ultra-grim, but allows us to break the TS_FORECASTREGRESSION
# logic into its own class which maintaining backwards compatibility of
# the public interface to AutoML
class AutoML(AutoMLGeneric):
    def __init__(self, **settings):
        super().__init__(**settings)
        if self._settings["task"] in TS_FORECASTREGRESSION:
            AutoMLTS.__init__(self, **settings)

    def fit(self, *args, **kwargs):
        # Is it in kwargs?
        task_name = kwargs.get("task")
        # Is it in args?
        if len(args) >= 6:
            task_name = args[5]
        # Is it in self._settings?
        task_name = task_name or self._settings["task"]
        task = tasks.task_factory(task_name)

        self.__class__ = task.AUTOML_CLASS
        self.fit(*args, **kwargs)


# Set the docstring for sphinx autodocs
AutoML.__init__.__doc__ = AutoMLGeneric.__init__.__doc__
AutoML.fit.__doc__ = AutoMLGeneric.fit.__doc__
