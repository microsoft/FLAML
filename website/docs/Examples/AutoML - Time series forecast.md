### Univariate time series

```python
# pip install flaml[ts_forecast]
import numpy as np
from flaml import AutoML
X_train = np.arange('2014-01', '2021-01', dtype='datetime64[M]')
y_train = np.random.random(size=72)
automl = AutoML()
automl.fit(X_train=X_train[:72],  # a single column of timestamp
           y_train=y_train,  # value for each timestamp
           period=12,  # time horizon to forecast, e.g., 12 months
           task='ts_forecast', time_budget=15,  # time budget in seconds
           log_file_name="test/ts_forecast.log",
          )
print(automl.predict(X_train[72:]))
```