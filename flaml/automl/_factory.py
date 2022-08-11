from .automl import AutoML as AutoMLGeneric
from .time_series import AutoMLTS
from ..data import TS_FORECASTREGRESSION


class AutoML(AutoMLGeneric):
    def __init__(self, **settings):
        super().__init__(**settings)

        # Hackity hack hack
        # The below is ultra-grim, but allows us to break the TS_FORECASTREGRESSION
        # logic into its own class which maintaining backwards compatibility of
        # the public interface to AutoML
        if self._settings["task"] in TS_FORECASTREGRESSION:
            automl_ts = AutoMLTS(**settings)
            self.__dict__ = automl_ts.__dict__
            self.__class__ = automl_ts.__class__


# Set the docstring for sphinx autodocs
AutoML.__init__.__doc__ = AutoMLGeneric.__init__.__doc__
