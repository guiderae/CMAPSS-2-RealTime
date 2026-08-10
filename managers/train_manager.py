"""
TrainManager: orchestrates ONE model.fit() run. Holds no class-level state
of its own -- the model lives on LSTMModel.model, not here.
"""
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import ModelConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from managers.graph_manager import GraphManager


class TrainManager:

    @classmethod
    def fit(cls, epochs=ModelConfig.DEFAULT_EPOCHS, learning_rate=ModelConfig.DEFAULT_LEARNING_RATE):
        """Reads DataPrepManager.X_train/y_train. Builds+compiles a fresh
        model via LSTMModel.create(). Trains WITHOUT validation_data --
        testing is a separate step now (see TestManager). Returns the
        response dict ready for jsonify."""
        # Guards against a Live Training background thread (see
        # realtime/process_realtime_training.py) using LSTMModel.model at
        # the same time -- without this, a sync train_model call during a
        # live-training run would race on the same shared object.
        if not LSTMModel.training_lock.acquire(blocking=False):
            return {'success': False, 'error': 'Training already in progress'}
        try:
            # Not present in CMAPSS-1 (no seed anywhere there); makes
            # /predict's headline number reproducible run-to-run for
            # verification.
            np.random.seed(ModelConfig.RANDOM_SEED)
            tf.random.set_seed(ModelConfig.RANDOM_SEED)

            X_train = DataPrepManager.X_train
            y_train = DataPrepManager.y_train

            input_shape = (X_train.shape[1], X_train.shape[2])
            LSTMModel.create(input_shape, learning_rate=learning_rate)

            classes = np.unique(y_train)
            weights = compute_class_weight(ModelConfig.CLASS_WEIGHT_STRATEGY, classes=classes, y=y_train)
            class_weight = dict(zip(classes.tolist(), weights))

            history = LSTMModel.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=ModelConfig.BATCH_SIZE,
                class_weight=class_weight,
                verbose=ModelConfig.VERBOSE
            )

            metric_key = ModelConfig.METRICS[0]
            plot_url = GraphManager.render_training_plot(history, metric_key=metric_key)

            return {
                'success': True,
                'message': 'Model trained successfully',
                'plot': plot_url,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history[metric_key][-1]
            }
        finally:
            LSTMModel.training_lock.release()
