"""
ProcessRealtimeData: streams one engine unit's data as Server-Sent Events,
computing a live failure-probability prediction once enough rows have
buffered to fill one LSTM input window (DataConfig.TIME_STEPS rows).
Reads LSTMModel.model and DataPrepManager.selected_sensors directly, same
convention as PredictManager/TestManager/TrainManager.

SSE message shapes:
  event: initialize
  data: {"unit": <int>, "sensors": [<str>, ...], "time_steps": <int>}

  event: update
  data: {"cycle": <int>, "sensors": {<str>: <float>, ...}, "prediction": <float|null>}

  event: jobfinished
  data: none
"""
import json
from collections import deque

import numpy as np

from config import DataConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from realtime.data_source_manager import DataSourceManager


class ProcessRealtimeData:

    def __init__(self, unit):
        self.unit = unit
        self.feature_columns = list(DataPrepManager.selected_sensors)
        self.buffer = deque(maxlen=DataConfig.TIME_STEPS)

    def process_points(self):
        """Generator of SSE-formatted strings. `unit` is a plain int
        resolved by the caller (the route) before this generator or the
        Response is constructed, so this generator never touches Flask's
        `request`/`session` -- avoids the classic 'working outside of
        request context' failure mode for streamed generators."""
        yield self.__sse('initialize', {
            'unit': self.unit,
            'sensors': self.feature_columns,
            'time_steps': DataConfig.TIME_STEPS
        })

        for cycle, row in DataSourceManager.stream_unit_rows(self.unit):
            # Predict from the buffer BEFORE appending the current row.
            # This ensures the window used to predict row i is exactly the
            # TIME_STEPS rows immediately before row i, matching
            # create_sequences()'s convention (window i..i+29 predicts the
            # label at row i+30) -- and never includes row i itself, even
            # when row i is the unit's actual failure row. Predicting
            # AFTER appending would silently reintroduce the off-by-one
            # bug fixed earlier in this project (a window that improperly
            # includes the failure row itself).
            prediction = self.__predict_if_ready()
            self.buffer.append(row)

            yield self.__sse('update', {
                'cycle': cycle,
                'sensors': dict(zip(self.feature_columns, (float(v) for v in row))),
                'prediction': prediction
            })

        yield self.__sse('jobfinished', None)

    def __predict_if_ready(self):
        """Predicts from the buffer as it stands BEFORE the current row is
        appended. Returns None until TIME_STEPS full rows of prior history
        exist, or if a Live Training background thread currently holds
        LSTMModel.training_lock (this row's prediction is simply skipped
        rather than blocking or erroring the whole stream)."""
        if len(self.buffer) < DataConfig.TIME_STEPS:
            return None
        window = np.array(self.buffer).reshape(1, DataConfig.TIME_STEPS, len(self.feature_columns))
        if not LSTMModel.training_lock.acquire(blocking=False):
            return None
        try:
            return float(LSTMModel.model.predict(window, verbose=0)[0, 0])
        finally:
            LSTMModel.training_lock.release()

    @staticmethod
    def __sse(event, payload):
        data = 'none' if payload is None else json.dumps(payload)
        return f"event: {event}\ndata: {data}\n\n"
