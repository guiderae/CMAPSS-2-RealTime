"""
DataFileManager: derives the list of engine units selectable for the
Prediction step's unit dropdown. Named/positioned to mirror the reference
demo's DataFileManager (which listed CSV file names on disk) -- CMAPSS-2
has one data file, not one file per unit, so this returns unit IDs instead.
"""
from managers.data_prep_manager import DataPrepManager


class DataFileManager:

    @staticmethod
    def get_selectable_units():
        """Units reserved for prediction only -- never trained or tested
        on (see DataPrepManager.get_unit_splits), derived from
        DataConfig's percentage-based split rather than a fixed list, so
        this scales with however many units the data file actually has."""
        all_units = DataPrepManager.get_unit_ids()
        _, _, predict_units = DataPrepManager.get_unit_splits(all_units)
        return sorted(predict_units)
