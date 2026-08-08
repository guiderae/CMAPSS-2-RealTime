"""
DataSourceManager: yields one already-scaled sensor row at a time for a
given engine unit, at a configurable pace, simulating real-time arrival.
Reuses DataPrepManager's already-fitted scaler/selected_sensors/
train_feature_means (NOT raw CSV line-by-line like the reference demo),
so streamed values are on the exact same footing as training/testing/
static-prediction data.
"""
import time

from config import RealtimeConfig
from managers.data_prep_manager import DataPrepManager


class DataSourceManager:

    @staticmethod
    def stream_unit_rows(unit, delay=RealtimeConfig.STREAM_DELAY_SECONDS):
        """Generator yielding (cycle, scaled_row) tuples for `unit`, one
        row at a time, sleeping `delay` seconds between rows. scaled_row
        is a 1D numpy array in the same column order as
        DataPrepManager.selected_sensors."""
        df = DataPrepManager.load_episodes()[unit]
        feature_columns = DataPrepManager.selected_sensors
        X_df = df[feature_columns].fillna(DataPrepManager.train_feature_means)
        X_scaled = DataPrepManager.scaler.transform(X_df.values)
        cycles = df['cycle'].values

        for cycle, row in zip(cycles, X_scaled):
            time.sleep(delay)
            yield int(cycle), row
