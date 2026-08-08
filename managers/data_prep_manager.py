"""
DataPrepManager: loads and prepares C-MAPSS data for the LSTM. Owns the
fitted scaler, per-feature training means, selected sensors, and the
prepared train/test sequence arrays as class-level state (replacing
CMAPSS-1's module globals of the same names, plus its data_storage dict).

load_episodes/create_sequences/compute_risk_labels/select_failure_relevant_sensors
are ported unchanged from CMAPSS-1/app.py -- select_failure_relevant_sensors
in particular was validated via 5-fold cross-validation (mean AUC 0.995) and
must not be altered.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import DataConfig


class DataPrepManager:
    scaler = None
    train_feature_means = None
    selected_sensors = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    @staticmethod
    def load_episodes():
        """Load DataConfig.DATA_FILE and split into one DataFrame per engine
        unit, each with a 'machine_status' column that's 0 except the final
        (failure) row."""
        # The data file is space delimited since all data are numeric.  r means raw string,
        # \s means any whitespace char (spaces, tabs, newlines), + means one or more times.
        raw = pd.read_csv(DataConfig.DATA_FILE, sep=r'\s+', header=None, names=DataConfig.COLUMNS)
        episodes = {}
        for unit in sorted(raw['unit'].unique()):
            edf = raw[raw['unit'] == unit].sort_values('cycle').reset_index(drop=True).copy()
            edf['machine_status'] = 0
            edf.loc[edf.index[-1], 'machine_status'] = 1
            episodes[int(unit)] = edf
        return episodes

    @staticmethod
    def create_sequences(data, target, time_steps=DataConfig.TIME_STEPS):
        """Create sliding-window sequences for the LSTM."""
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(target[i + time_steps])
        return np.array(X), np.array(y)

    @staticmethod
    def compute_risk_labels(df, risk_window=DataConfig.RISK_WINDOW):
        """Label each row 1 if a failure (machine_status == 1) occurs within
        `risk_window` rows ahead (inclusive of the failure row itself), else 0."""
        status = df['machine_status'].values
        n = len(status)
        risk = np.zeros(n, dtype=int)
        for idx in np.where(status == 1)[0]:
            start = max(0, idx - risk_window + 1)
            risk[start:idx + 1] = 1
        return risk

    @staticmethod
    def select_failure_relevant_sensors(dfs, candidate_columns, top_n=DataConfig.TOP_N_SENSORS,
                                         window=DataConfig.TIME_STEPS):
        """Rank candidate sensors by how consistently they shift from their
        OWN per-episode baseline (first `window` rows) to their own
        pre-failure window (last `window` rows), across every episode in
        `dfs`, and return the top_n. This favors sensors that reliably
        deviate near failure over sensors that are just naturally noisy."""
        scores = {}
        for col in candidate_columns:
            local_zs = []
            for df in dfs:
                base = df[col].iloc[:window]
                pre = df[col].iloc[-window:]
                base_std = base.std()
                if df[col].isnull().all() or base_std == 0 or pd.isna(base_std):
                    continue
                local_zs.append((pre.mean() - base.mean()) / base_std)
            if len(local_zs) == len(dfs):  # only rank sensors valid in every episode
                scores[col] = np.median(np.abs(local_zs))
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[:top_n]

    @classmethod
    def prepare(cls, train_units=None, test_units=None):
        """Orchestrates load_episodes -> compute_risk_labels ->
        select_failure_relevant_sensors -> fit StandardScaler -> build
        per-unit sequences (never spanning two units). Sets
        cls.scaler/train_feature_means/selected_sensors/X_train/y_train/
        X_test/y_test. Returns the response dict ready for jsonify."""
        train_units = list(train_units) if train_units is not None else list(DataConfig.TRAIN_UNIT_RANGE)
        test_units = list(test_units) if test_units is not None else list(DataConfig.TEST_UNIT_RANGE)

        all_episodes = cls.load_episodes()
        train_dfs = [all_episodes[u] for u in train_units]
        test_dfs = [all_episodes[u] for u in test_units]

        for df in train_dfs + test_dfs:
            df['risk_label'] = cls.compute_risk_labels(df)

        candidate_columns = [c for c in DataConfig.COLUMNS if c not in ('unit', 'cycle')]
        cls.selected_sensors = cls.select_failure_relevant_sensors(train_dfs, candidate_columns)
        feature_columns = cls.selected_sensors

        train_concat = pd.concat(train_dfs, ignore_index=True)
        cls.train_feature_means = train_concat[feature_columns].mean()
        X_train_all = train_concat[feature_columns].fillna(cls.train_feature_means).values

        cls.scaler = StandardScaler()
        cls.scaler.fit(X_train_all)

        def build_sequences(dfs):
            """Build LSTM sequences per-unit, then concatenate the sequences
            (not the raw rows) so no window spans two different engines."""
            X_parts, y_parts = [], []
            for df in dfs:
                X_df = df[feature_columns].fillna(cls.train_feature_means)
                X_scaled = cls.scaler.transform(X_df.values)
                X_seq, y_seq = cls.create_sequences(X_scaled, df['risk_label'].values)
                if len(X_seq):
                    X_parts.append(X_seq)
                    y_parts.append(y_seq)
            if not X_parts:
                return (np.empty((0, DataConfig.TIME_STEPS, len(feature_columns))), np.empty((0,)))
            return np.concatenate(X_parts), np.concatenate(y_parts)

        cls.X_train, cls.y_train = build_sequences(train_dfs)
        cls.X_test, cls.y_test = build_sequences(test_dfs)

        return {
            'success': True,
            'message': 'Data prepared successfully',
            'train_shape': cls.X_train.shape,
            'test_shape': cls.X_test.shape,
            'features_after_pca': len(cls.selected_sensors)  # kept for UI compatibility; no PCA is used
        }

    @classmethod
    def reset(cls):
        cls.scaler = None
        cls.train_feature_means = None   # fixes a CMAPSS-1 bug: reset() never cleared this
        cls.selected_sensors = None
        cls.X_train = cls.y_train = cls.X_test = cls.y_test = None
