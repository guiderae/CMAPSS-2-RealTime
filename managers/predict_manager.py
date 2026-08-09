"""
PredictManager: runs the trained model against the held-out prediction
unit. Holds no class-level state of its own -- reads
DataPrepManager.scaler/train_feature_means/selected_sensors and
LSTMModel.model.
"""
from config import DataConfig, PlotConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from managers.graph_manager import GraphManager


class PredictManager:

    @classmethod
    def predict(cls, unit=None):
        """Loads the given unit fresh (defaults to the first unit in the
        predict-allocation group -- see DataPrepManager.get_unit_splits --
        never a train/test unit either way). Returns the response dict
        ready for jsonify, or {'success': False, 'error': ...} if the
        unit isn't in the dataset or there's insufficient data."""
        if unit is None:
            _, _, predict_units = DataPrepManager.get_unit_splits(DataPrepManager.get_unit_ids())
            unit = predict_units[0] if predict_units else None

        episodes = DataPrepManager.load_episodes()
        if unit not in episodes:
            return {'success': False, 'error': f'Unit {unit} not found in dataset'}
        df = episodes[unit]
        risk_label = DataPrepManager.compute_risk_labels(df)

        feature_columns = DataPrepManager.selected_sensors
        X_pred_df = df[feature_columns].fillna(DataPrepManager.train_feature_means)
        X_pred_scaled = DataPrepManager.scaler.transform(X_pred_df.values)

        if len(X_pred_scaled) < DataConfig.TIME_STEPS + 1:
            return {'success': False, 'error': 'Insufficient data for prediction'}

        # Build every sliding window in the unit's life (same alignment used
        # in training: window i..i+29 predicts the label at row i+30) so we
        # get a full risk-probability curve, not a single snapshot. The
        # final window's target is the label of the last row (the known
        # failure row), so its prediction is the headline number.
        X_pred_seq, _ = DataPrepManager.create_sequences(X_pred_scaled, risk_label)
        predictions = LSTMModel.model.predict(X_pred_seq).flatten()
        prediction = float(predictions[-1])
        failure_idx = len(X_pred_scaled) - 1

        plot_url = GraphManager.render_prediction_plot(
            X_pred_scaled, feature_columns, predictions, failure_idx,
            DataConfig.TIME_STEPS, unit
        )

        return {
            'success': True,
            'prediction': prediction,
            'plot': plot_url,
            'status': 'Failure Risk' if prediction > PlotConfig.PROBABILITY_THRESHOLD else 'Normal Operation'
        }
