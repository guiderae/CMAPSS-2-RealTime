"""
The LSTM model definition and the one shared trained instance.

This is deliberately NOT one of the 5 Manager classes (DataPrepManager,
TrainManager, TestManager, PredictManager, GraphManager) -- it's the model
artifact itself. TrainManager builds/fits it, TestManager evaluates it, and
PredictManager runs predictions with it; all three read LSTMModel.model
directly rather than TrainManager owning the model on their behalf.
"""
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    try:
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.optimizers import Adam
    except ImportError:
        raise ImportError("TensorFlow/Keras not found. Please install with: pip install tensorflow")

from config import ModelConfig


class LSTMModel:
    model = None   # class-level Keras model -- the ONE shared trained instance

    @staticmethod
    def build(input_shape):
        """Builds (does not compile) the Sequential LSTM architecture."""
        return Sequential([
            LSTM(ModelConfig.LSTM_UNITS_1, return_sequences=True, input_shape=input_shape),
            Dropout(ModelConfig.DROPOUT_RATE_1),
            LSTM(ModelConfig.LSTM_UNITS_2, return_sequences=False),
            Dropout(ModelConfig.DROPOUT_RATE_2),
            Dense(ModelConfig.DENSE_UNITS, activation=ModelConfig.DENSE_ACTIVATION),
            Dense(1, activation=ModelConfig.OUTPUT_ACTIVATION)
        ])

    @classmethod
    def create(cls, input_shape, learning_rate=ModelConfig.DEFAULT_LEARNING_RATE):
        """Builds via build(), compiles, sets cls.model, and returns it.
        Called once at the start of each training run -- a fresh model each
        time, matching CMAPSS-1's behavior of rebuilding on every
        /train_model call."""
        cls.model = cls.build(input_shape)
        cls.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=ModelConfig.LOSS,
            metrics=ModelConfig.METRICS
        )
        return cls.model

    @classmethod
    def reset(cls):
        cls.model = None
