"""
TestManager: a pure post-hoc evaluator. Holds no class-level state of its
own -- reads LSTMModel.model and DataPrepManager.X_test/y_test, produces a
one-shot result each call.
"""
from config import ModelConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from managers.graph_manager import GraphManager


class TestManager:

    @classmethod
    def evaluate(cls):
        """Runs the already-trained model against the held-out test units.
        Returns the response dict ready for jsonify."""
        X_test = DataPrepManager.X_test
        y_test = DataPrepManager.y_test

        test_loss, test_accuracy = LSTMModel.model.evaluate(X_test, y_test, verbose=ModelConfig.VERBOSE)
        y_pred_prob = LSTMModel.model.predict(X_test).flatten()

        plot_url = GraphManager.render_test_plot(y_test, y_pred_prob)

        return {
            'success': True,
            'message': 'Model tested successfully',
            'plot': plot_url,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
