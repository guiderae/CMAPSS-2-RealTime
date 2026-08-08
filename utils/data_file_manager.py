"""
DataFileManager: derives the list of engine units selectable for the
Prediction step's unit dropdown. Named/positioned to mirror the reference
demo's DataFileManager (which listed CSV file names on disk) -- CMAPSS-2
has one data file, not one file per unit, so this returns unit IDs instead.
"""
from config import DataConfig


class DataFileManager:

    @staticmethod
    def get_selectable_units():
        """Units the model never trained on: the held-out test range plus
        the dedicated prediction unit. Derived from existing DataConfig
        constants, never a separately hardcoded list."""
        units = list(DataConfig.TEST_UNIT_RANGE) + [DataConfig.PREDICT_UNIT]
        return sorted(units)
