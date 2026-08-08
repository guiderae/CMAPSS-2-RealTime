"""
All hardcoded values for the CMAPSS-2 app live here, grouped by concern.
Each class holds class-level (static) attributes only -- nothing here is
ever instantiated, they're accessed directly as e.g. DataConfig.TIME_STEPS.
"""


class DataConfig:
    """Dataset location, shape, and the fixed train/test/predict unit split.

    NASA's train_FD001.txt already contains all 100 units as complete
    run-to-failure trajectories -- this app never touches the official
    test_FD001.txt/RUL_FD001.txt (a different, truncated-before-failure set
    of engines used for NASA's own benchmark scoring). Instead,
    TRAIN_UNIT_RANGE/TEST_UNIT_RANGE/PREDICT_UNIT slice all three of our
    own train/test/predict groups out of the same 100-unit file by unit ID.
    """
    DATA_FILE = 'data/train_FD001.txt'
    COLUMNS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]

    TRAIN_UNIT_RANGE = range(1, 71)    # 70 units
    TEST_UNIT_RANGE = range(71, 91)    # 20 units
    PREDICT_UNIT = 91                  # single held-out unit, never trained on

    TIME_STEPS = 30    # well under the shortest unit's 128-cycle lifespan
    RISK_WINDOW = 30   # "failure within the next 30 cycles" -- standard in RUL literature
    TOP_N_SENSORS = 10


class ModelConfig:
    """LSTM architecture, training hyperparameters, and their UI-request defaults."""
    LSTM_UNITS_1 = 64
    LSTM_UNITS_2 = 32
    DROPOUT_RATE_1 = 0.2
    DROPOUT_RATE_2 = 0.2
    DENSE_UNITS = 16
    DENSE_ACTIVATION = 'relu'
    OUTPUT_ACTIVATION = 'sigmoid'

    LOSS = 'binary_crossentropy'
    METRICS = ['accuracy']
    CLASS_WEIGHT_STRATEGY = 'balanced'

    DEFAULT_EPOCHS = 20
    DEFAULT_LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    VERBOSE = 1

    # Not present in CMAPSS-1 (no seed anywhere there, so /predict's headline
    # number wasn't reproducible run-to-run). Added here so behavior is
    # deterministic and verifiable -- see TrainManager.fit().
    RANDOM_SEED = 42


class PlotConfig:
    """Figure sizes and shared plot styling constants."""
    TRAIN_FIGSIZE = (12, 4)
    TEST_FIGSIZE = (8, 6)
    PREDICT_FIGSIZE = (12, 8)

    PROBABILITY_THRESHOLD = 0.5
    THRESHOLD_LINE_COLOR = 'orange'
    FAILURE_LINE_COLOR = 'black'
    PREDICTION_LINE_COLOR = 'red'


class RealtimeConfig:
    """Constants for the Phase 3 streaming/SSE prediction mode."""
    STREAM_DELAY_SECONDS = 0.1   # pace between simulated sensor readings
    MAX_CHART_POINTS = 50       # client-side Plotly rolling-window trim, ie start scrolling after MAX_CHART_POINT time units


class FlaskConfig:
    SECRET_KEY = 'your-secret-key-change-this'
    PORT = 8082
    DEBUG = True
    THREADED = True   # required so the long-lived SSE connection doesn't block other requests
