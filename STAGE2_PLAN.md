# Stage 2: OOP refactor of the CMAPSS ML pipeline (CMAPSS-1 → CMAPSS-2)

## Context

`CMAPSS-1` is a working proof-of-concept Flask app (confirmed working end-to-end,
including plots) that predicts turbofan-engine failure from NASA C-MAPSS sensor
data. Its `app.py` is a single 382-line file where Flask route functions contain
all the ML logic directly (data loading, labeling, scaling, model training,
prediction, and matplotlib plotting), and shared state between requests lives in
five bare module-level globals (`data_storage`, `model`, `scaler`,
`train_feature_means`, `selected_sensors`).

This is Stage 2 of a 3-stage plan: take the proven POC and rewrite it as a proper
object-oriented web application with the same functional behavior, so it's
maintainable and extensible going forward. Three explicit requirements drove this:

1. `app.py` becomes a pure controller - route functions contain no business logic,
   only calls into 5 Manager classes: **DataPrepManager, TrainManager, TestManager,
   PredictManager, GraphManager**.
2. No module-level global variables - cross-request state lives as class-level
   (static) attributes on whichever Manager class produces it.
3. Every hardcoded value (file paths, column names, unit ranges, model
   hyperparameters, thresholds, port number, etc.) moves into `config.py`.

One functional addition came out of clarifying requirement 1: **testing must be a
separate, explicitly-triggered step** (`POST /test_model`), not bundled into
`/train_model` as before - the rationale being that a user should be able to
iterate on training (re-run with different epochs/learning rate, eyeball the
loss/accuracy curve) without paying for a held-out evaluation pass every time, and
only test once satisfied with training. This means removing `validation_data` from
`model.fit()` and adding a genuinely new endpoint + a 4th UI wizard step.

A later clarification added a 6th class: **`LSTMModel`** (not a "Manager") - the
model architecture/definition and the one shared trained instance, held as a
static `model` attribute directly accessible from `TrainManager`, `TestManager`,
and `PredictManager`, rather than `TrainManager` owning the model itself.

Target location: **new project `/Users/eliguidera/PycharmProjects/CMAPSS-2/`**,
leaving `CMAPSS-1` untouched as the known-good reference.

## Directory layout

```
CMAPSS-2/
├── venv/                          # fresh venv, same pinned deps as CMAPSS-1
├── requirements.txt                # copied verbatim from CMAPSS-1
├── config.py                       # all constants, as small grouped classes
├── lstm_model.py                   # LSTMModel - the model definition + shared instance
├── app.py                          # thin Flask controller only
├── managers/
│   ├── __init__.py
│   ├── data_prep_manager.py
│   ├── train_manager.py
│   ├── test_manager.py
│   ├── predict_manager.py
│   └── graph_manager.py
├── data/
│   └── train_FD001.txt             # copied from CMAPSS-1/data/ (the only file app.py reads)
└── templates/
    └── index.html                  # copied from CMAPSS-1, then patched with a 4th step
```

`LSTMModel` lives at the project root, not inside `managers/`, since it isn't a
workflow-step orchestrator like the 5 Manager classes - it's the model artifact
itself (architecture + the one shared trained instance), which `TrainManager`,
`TestManager`, and `PredictManager` all need direct read access to.

Not carried over: `data/test_FD001.txt`, `data/RUL_FD001.txt`, and `validation/`
(Stage 1's cross-validation script) - none are read by the running app; only
`train_FD001.txt` is used, and the app manufactures its own train/test/predict
split from the 100 units inside it.

**Note on the data split (NASA's naming is misleading here):** `train_FD001.txt`
already contains all 100 units as complete run-to-failure trajectories. NASA's
official `test_FD001.txt`/`RUL_FD001.txt` are a *different* set of engines,
truncated before failure, for the official benchmark's separate scoring - our app
never reads them. Instead, `DataConfig.TRAIN_UNIT_RANGE` / `TEST_UNIT_RANGE` /
`PREDICT_UNIT` slice all three of our train/test/predict groups out of the same
100-unit `train_FD001.txt` by unit ID. So the train/test/predict allocation is
already fully config-driven - only `train_FD001.txt` needs to be present in `data/`
for `config.py` to control all three splits.

## `config.py`

Four small classes, grouped by concern, all class-level attributes (no
instantiation needed) so managers read them as `DataConfig.TIME_STEPS` etc.:

```python
class DataConfig:
    DATA_FILE = 'data/train_FD001.txt'
    COLUMNS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
    TRAIN_UNIT_RANGE = range(1, 71)
    TEST_UNIT_RANGE = range(71, 91)
    PREDICT_UNIT = 91
    TIME_STEPS = 30
    RISK_WINDOW = 30
    TOP_N_SENSORS = 10

class ModelConfig:
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
    RANDOM_SEED = 42   # new - see "Determinism" below

class PlotConfig:
    TRAIN_FIGSIZE = (12, 4)
    TEST_FIGSIZE = (8, 6)
    PREDICT_FIGSIZE = (12, 8)
    PROBABILITY_THRESHOLD = 0.5
    THRESHOLD_LINE_COLOR = 'orange'
    FAILURE_LINE_COLOR = 'black'
    PREDICTION_LINE_COLOR = 'red'

class FlaskConfig:
    SECRET_KEY = 'your-secret-key-change-this'
    PORT = 8082
    DEBUG = True
```

`history.history[...]` lookups use `ModelConfig.METRICS[0]` rather than the
literal string `'accuracy'`, per the letter of "all hardcoded values."

## Manager classes

All five live under `managers/`. Class-level attributes replace the old module
globals; each manager owns the state it produces. Managers are allowed to call
other managers (e.g. `TrainManager` reads `DataPrepManager`'s data,
`PredictManager` calls `GraphManager`) - that's normal service-layer composition,
not a violation of "manager owns its logic."

**Key design decision:** each primary manager's top-level method (`prepare()`,
`fit()`, `evaluate()`, `predict()`) returns a **complete, ready-to-`jsonify` dict**
(including `plot`, computed by calling `GraphManager` internally). This keeps
`app.py` from ever touching a numpy array, a matplotlib figure, or response-shape
decisions - routes become guard → manager call → `jsonify(result)` → except,
nothing else.

### `DataPrepManager`
```python
class DataPrepManager:
    scaler = None
    train_feature_means = None
    selected_sensors = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    @staticmethod
    def load_episodes() -> dict[int, pd.DataFrame]: ...
    @staticmethod
    def create_sequences(data, target, time_steps=DataConfig.TIME_STEPS): ...
    @staticmethod
    def compute_risk_labels(df, risk_window=DataConfig.RISK_WINDOW): ...
    @staticmethod
    def select_failure_relevant_sensors(dfs, candidate_columns,
                                         top_n=DataConfig.TOP_N_SENSORS,
                                         window=DataConfig.TIME_STEPS): ...

    @classmethod
    def prepare(cls, train_units=None, test_units=None) -> dict:
        """Defaults to DataConfig.TRAIN_UNIT_RANGE/TEST_UNIT_RANGE if not
        given (optional params purely for testability). Orchestrates
        load_episodes -> compute_risk_labels -> select_failure_relevant_sensors
        -> fit StandardScaler -> build per-unit sequences (never spanning two
        units). Sets cls.scaler/train_feature_means/selected_sensors/X_train/
        y_train/X_test/y_test. Returns:
        {'success': True, 'message': 'Data prepared successfully',
         'train_shape': cls.X_train.shape, 'test_shape': cls.X_test.shape,
         'features_after_pca': len(cls.selected_sensors)}
        ('features_after_pca' key name kept verbatim - index.html's JS reads
        it; no PCA is actually used, same as in CMAPSS-1.)"""

    @classmethod
    def reset(cls):
        cls.scaler = None
        cls.train_feature_means = None   # fixes a CMAPSS-1 bug: reset() never cleared this
        cls.selected_sensors = None
        cls.X_train = cls.y_train = cls.X_test = cls.y_test = None
```

These four static methods are the exact, empirically-validated logic from
`CMAPSS-1/app.py` (`load_episodes`, `create_sequences`, `compute_risk_labels`,
`select_failure_relevant_sensors`) - ported unchanged. `select_failure_relevant_sensors`
in particular was validated via 5-fold cross-validation (mean AUC 0.995) and must
not be altered.

### `LSTMModel` (not a Manager - the model artifact itself)

```python
class LSTMModel:
    model = None   # class-level Keras model, the ONE shared trained instance --
                    # read directly by TrainManager (to fit it), TestManager
                    # (to evaluate it), and PredictManager (to predict with it)

    @staticmethod
    def build(input_shape) -> Sequential:
        """Builds (uncompiled) the LSTM(ModelConfig.LSTM_UNITS_1)->Dropout->
        LSTM(ModelConfig.LSTM_UNITS_2)->Dropout->Dense(ModelConfig.DENSE_UNITS,
        ModelConfig.DENSE_ACTIVATION)->Dense(1,ModelConfig.OUTPUT_ACTIVATION)
        architecture -- identical structure to CMAPSS-1's create_model(),
        using ModelConfig constants instead of literals."""

    @classmethod
    def create(cls, input_shape, learning_rate=ModelConfig.DEFAULT_LEARNING_RATE) -> Sequential:
        """Builds via build(), compiles with Adam(learning_rate),
        ModelConfig.LOSS, ModelConfig.METRICS. Sets cls.model. Returns
        cls.model. Called by TrainManager.fit() at the start of each
        training run (a fresh model each time, matching CMAPSS-1's
        behavior of rebuilding on every /train_model call)."""

    @classmethod
    def reset(cls):
        cls.model = None
```

### `TrainManager`
```python
class TrainManager:
    # No class-level state of its own -- the model lives on LSTMModel.model,
    # not here. TrainManager is purely the orchestration of ONE fit() run.

    @classmethod
    def fit(cls, epochs=ModelConfig.DEFAULT_EPOCHS,
            learning_rate=ModelConfig.DEFAULT_LEARNING_RATE) -> dict:
        """Reads DataPrepManager.X_train/y_train. Calls
        LSTMModel.create((X_train.shape[1], X_train.shape[2]), learning_rate)
        to build+compile+store the model on LSTMModel.model. Computes
        balanced class_weight from y_train. Calls LSTMModel.model.fit(
        X_train, y_train, epochs=, batch_size=ModelConfig.BATCH_SIZE,
        class_weight=, verbose=ModelConfig.VERBOSE) -- NO validation_data
        (testing is now a separate step). Calls
        GraphManager.render_training_plot(history) and returns:
        {'success': True, 'message': 'Model trained successfully',
         'plot': plot_url, 'final_loss': history.history['loss'][-1],
         'final_accuracy': history.history[ModelConfig.METRICS[0]][-1]}"""
```

### `TestManager`
```python
class TestManager:
    @classmethod
    def evaluate(cls) -> dict:
        """Reads LSTMModel.model and DataPrepManager.X_test/y_test.
        Runs model.evaluate(X_test, y_test, verbose=ModelConfig.VERBOSE) for
        (test_loss, test_accuracy), and model.predict(X_test) for the
        probability array. Calls
        GraphManager.render_test_plot(y_test, y_pred_prob) and returns:
        {'success': True, 'message': 'Model tested successfully',
         'plot': plot_url, 'test_loss': float(test_loss),
         'test_accuracy': float(test_accuracy)}"""
```

Holds no class-level state - it's a pure evaluator over the other two managers'
state, produces a one-shot result each call.

**Test plot design decision:** a post-hoc `evaluate()` call has no epoch history to
plot (that only exists during `fit()`, and `validation_data` is gone from `fit()`
now). `GraphManager.render_test_plot` will instead show the **predicted-probability
distribution on the test set, split by true label** (e.g. two overlaid histograms
or a strip plot for true-0 vs. true-1 rows) with a vertical line at
`PlotConfig.PROBABILITY_THRESHOLD` - this is the only artifact `evaluate()`
naturally produces. This is a new visualization, not a port of an existing one;
flagging it here explicitly since the original requirements didn't specify what
the test plot should contain.

### `PredictManager`
```python
class PredictManager:
    @classmethod
    def predict(cls) -> dict:
        """Loads DataConfig.PREDICT_UNIT fresh via DataPrepManager.load_episodes()
        (never in train/test units). Computes risk_label via
        DataPrepManager.compute_risk_labels(). Scales
        DataPrepManager.selected_sensors columns via DataPrepManager.scaler
        + .train_feature_means.fillna(). If fewer than TIME_STEPS+1 rows:
        return {'success': False, 'error': 'Insufficient data for prediction'}.
        Otherwise builds sequences via DataPrepManager.create_sequences(),
        runs LSTMModel.model.predict(...), takes the LAST prediction
        (target = the true failure row) as the headline number. Calls
        GraphManager.render_prediction_plot(...) and returns:
        {'success': True, 'prediction': float, 'plot': plot_url,
         'status': 'Failure Risk' if prediction > PlotConfig.PROBABILITY_THRESHOLD
                    else 'Normal Operation'}"""
```

### `GraphManager`
Pure rendering, no dataset-specific knowledge - only imports `PlotConfig`, takes
plain arrays/scalars as arguments, returns base64 PNG strings. All `@staticmethod`s:

```python
class GraphManager:
    @staticmethod
    def _fig_to_base64(fig) -> str: ...   # savefig->BytesIO->b64encode->plt.close, same as old plot_to_base64

    @staticmethod
    def render_training_plot(history) -> str:
        """1x2, PlotConfig.TRAIN_FIGSIZE. Loss + accuracy curves,
        TRAINING ONLY now (no val_* -- validation_data was removed from
        fit()). This is an intentional visual change from CMAPSS-1's
        training plot, which showed validation curves when a test set was
        present."""

    @staticmethod
    def render_test_plot(y_true, y_pred_prob) -> str:
        """1x1 or similar, PlotConfig.TEST_FIGSIZE. New plot -- see
        TestManager section above."""

    @staticmethod
    def render_prediction_plot(X_pred_scaled, feature_columns, predictions,
                                failure_idx, time_steps, predict_unit) -> str:
        """2x1, PlotConfig.PREDICT_FIGSIZE. Top: selected sensor curves +
        failure-cycle vline. Bottom: probability-over-time + threshold hline
        + failure-cycle vline. Same figure CMAPSS-1's /predict already
        builds, just parameterized instead of closing over module globals."""
```

## `app.py` - 6 routes, fully thin

```python
from flask import Flask, render_template, request, jsonify, session
from config import ModelConfig, FlaskConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from managers.train_manager import TrainManager
from managers.test_manager import TestManager
from managers.predict_manager import PredictManager

app = Flask(__name__)
app.secret_key = FlaskConfig.SECRET_KEY


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prepare_data', methods=['POST'])
def prepare_data_route():
    try:
        result = DataPrepManager.prepare()
        session['data_prepared'] = True
        session['train_shape'] = result['train_shape']
        session['test_shape'] = result['test_shape']
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/train_model', methods=['POST'])
def train_model_route():
    if not session.get('data_prepared'):
        return jsonify({'success': False, 'error': 'Data not prepared'})
    try:
        epochs = int(request.json.get('epochs', ModelConfig.DEFAULT_EPOCHS))
        learning_rate = float(request.json.get('learning_rate', ModelConfig.DEFAULT_LEARNING_RATE))
        result = TrainManager.fit(epochs=epochs, learning_rate=learning_rate)
        session['model_trained'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/test_model', methods=['POST'])
def test_model_route():
    if not session.get('model_trained'):
        return jsonify({'success': False, 'error': 'Model not trained'})
    try:
        result = TestManager.evaluate()
        session['model_tested'] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict', methods=['POST'])
def predict_route():
    if not session.get('model_trained'):
        return jsonify({'success': False, 'error': 'Model not trained'})
    try:
        result = PredictManager.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/reset')
def reset_route():
    try:
        session.clear()
        DataPrepManager.reset()
        LSTMModel.reset()
        return jsonify({'success': True, 'message': 'Application reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(port=FlaskConfig.PORT, debug=FlaskConfig.DEBUG)
```

Notes:
- `/predict`'s guard stays on `session.get('model_trained')`, NOT a "tested" flag -
  testing is diagnostic, not a prerequisite for prediction, and CMAPSS-1's
  `/predict` contract must stay compatible.
- `session['train_shape']`/`test_shape` now come from the dict `DataPrepManager.prepare()`
  returns rather than being read off a local variable - same values, same shapes
  (plain tuples), no behavior change.
- Added a try/except to `/reset` for consistency (CMAPSS-1's `reset()` had none) -
  a small robustness improvement, not a behavior change under normal operation.
- Import list only pulls in what routes actually touch (`ModelConfig` for the two
  request-body defaults, `FlaskConfig` for secret key/port/debug) - everything
  else stays inside the managers.

## Session flags vs. Manager-owned class state

Two lifetimes, cleanly separated (same overall concurrency model as CMAPSS-1 -
this refactor relocates state, it doesn't redesign multi-tenancy):

- **`session`** (Flask, per-browser cookie): only booleans/shapes for route
  guarding - `data_prepared`, `train_shape`, `test_shape`, `model_trained`,
  `model_tested`. Never holds arrays or the model.
- **Manager/model class attributes** (process memory, single active run): the
  actual fitted artifacts - `DataPrepManager.{scaler,train_feature_means,
  selected_sensors,X_train,y_train,X_test,y_test}` and `LSTMModel.model`.
  `TrainManager` itself holds no state - it's pure orchestration of a `fit()`
  call, reading `DataPrepManager`'s data and writing to `LSTMModel.model`.

`/reset` clears both layers via `DataPrepManager.reset()` and `LSTMModel.reset()`.
`TrainManager`/`TestManager`/`PredictManager`/`GraphManager` need no `reset()` -
none of them hold persistent state of their own.

## Determinism (flagged addition beyond pure refactor)

CMAPSS-1 has no random seed anywhere, so `/predict`'s headline number is not
reproducible run-to-run (Keras weight init + class-weighted training are
unseeded). Since Stage 2's verification involves comparing CMAPSS-2's behavior
against CMAPSS-1, I'm adding `np.random.seed(ModelConfig.RANDOM_SEED)` and
`tf.random.set_seed(ModelConfig.RANDOM_SEED)` at the top of `TrainManager.fit()`,
before it calls `LSTMModel.create()` (so the fresh weight initialization is
seeded). This costs nothing functionally and makes manual verification meaningfully
checkable, but it is a behavior addition, not something the stated requirements
asked for - flagging it explicitly rather than adding it silently.

## `templates/index.html` diff (new Test Model step)

Copied from CMAPSS-1 verbatim, then these surgical changes only:

1. **`.wizard-steps`**: insert a new step div between step2 (Training) and the
   Prediction step:
   ```html
   <div class="step" id="step3">
       <h3>🧪 Test Model</h3>
       <p>Evaluate on held-out test units</p>
   </div>
   ```
   Renumber the existing Prediction step div `id="step3"` → `id="step4"`.

2. **`.content`**: insert a new content div between `content2` and the Prediction
   content div:
   ```html
   <div class="step-content" id="content3">
       <h2>Test Model</h2>
       <p>Evaluate the trained model against the held-out test units (engines 71–90).</p>
       <button onclick="testModel()">Test Model</button>
       <div class="loading" id="loading3">
           <div class="spinner"></div>
           <p>Testing model...</p>
       </div>
       <div id="test-results"></div>
   </div>
   ```
   Renumber the existing Prediction content div `id="content3"` → `id="content4"`,
   and **inside it**, its own `id="loading3"` → `id="loading4"` (to avoid
   colliding with the new step's `loading3`).

3. **`<script>`**:
   - Add `let modelTested = false;` next to `dataReady`/`modelTrained`.
   - `nextStep()`: bound changes from `currentStep < 3` to `currentStep < 4`.
   - `showStep()`: insert a branch for step 3 before the final `else` (which now
     applies only to step 4):
     ```js
     } else if (stepNum === 3) {
         nextBtn.style.display = modelTested ? 'block' : 'none';
     } else {
         nextBtn.style.display = 'none';
     }
     ```
   - New `testModel()` function, structurally mirroring `trainModel()`: calls
     `showLoading(3)`, `fetch('/test_model', {method:'POST', headers:{'Content-Type':'application/json'}})`,
     on success sets `modelTested = true`, `updateStepStatus(3, 'completed')`,
     populates `#test-results` with a metrics grid (`test_loss`, `test_accuracy`)
     and the plot image, calls `showStep(3)`; on failure/error shows an alert -
     same pattern as `trainModel()`.
   - `makePrediction()`: change `showLoading(3)`/`hideLoading(3)` →
     `showLoading(4)`/`hideLoading(4)`, and `updateStepStatus(3, 'completed')` →
     `updateStepStatus(4, 'completed')` (it now refers to the renumbered step 4).
   - `resetApplication()`: add `modelTested = false;` and
     `document.getElementById('test-results').innerHTML = '';` alongside the
     existing resets.

Everything else (CSS, header text, `prepareData()`, `resetApplication()`'s fetch
logic) stays untouched.

## Setup steps

1. Create `/Users/eliguidera/PycharmProjects/CMAPSS-2/` with the directory layout
   above.
2. Copy `templates/index.html` and `requirements.txt` from CMAPSS-1, then patch
   `index.html` per the diff above.
3. Copy `data/train_FD001.txt` from CMAPSS-1.
4. Create fresh venv, install from `requirements.txt` (same versions as CMAPSS-1:
   Flask 2.3.3, pandas 2.0.3, numpy 1.24.3, scikit-learn 1.3.0, tensorflow 2.15.0,
   matplotlib 3.7.2).
5. Write `config.py`, then `lstm_model.py`, `managers/graph_manager.py`, and
   `managers/data_prep_manager.py` (no dependencies on each other), then
   `managers/train_manager.py` (depends on DataPrepManager + LSTMModel +
   GraphManager), then `managers/test_manager.py` and `managers/predict_manager.py`
   (depend on DataPrepManager + LSTMModel + GraphManager), then `app.py`.

## Verification

Manual smoke test against a live server (matching the rigor used to validate
CMAPSS-1), on port 8082 with CMAPSS-1's server stopped to avoid a collision:

1. `GET /` - 200, wizard renders with **4** visible steps.
2. `POST /prepare_data` - `train_shape`/`test_shape`/`features_after_pca` should
   be **exactly** reproducible run-to-run (no randomness in data loading/labeling/
   sensor-selection/scaling) - compare against CMAPSS-1's `(12030, 30, 10)` /
   `(3650, 30, 10)` / `10` from the last verified run.
3. `POST /train_model` - confirm `final_loss`/`final_accuracy` present and
   plausible; with the new fixed seed, this becomes exactly reproducible across
   repeated CMAPSS-2 runs (not necessarily identical to CMAPSS-1's unseeded
   historical run).
4. `POST /test_model` (new) - confirm `test_loss`/`test_accuracy` returned and
   distinct from anything in the training response, proving it's a genuinely
   separate evaluation pass, not leftover `validation_data` state.
5. `POST /predict` - confirm shape (`prediction` float 0–1, `status` string,
   non-empty `plot`); with unit 91 held out identically to CMAPSS-1, this should
   again predict "Failure Risk" at high confidence, matching CMAPSS-1's 0.9996
   result within reason (exact match not guaranteed even with a seed, due to
   TF's own internal nondeterminism on some ops, but should be close and
   consistently >0.5).
6. `GET /reset` - `{success:true}`; then confirm `/train_model` immediately after
   returns `{'success': False, 'error': 'Data not prepared'}` (proves
   `DataPrepManager.reset()` worked); re-run `/prepare_data` → `/predict` without
   `/train_model` in between and confirm it correctly errors (proves
   `LSTMModel.reset()` cleared `model`).
7. Full sequence test: `/prepare_data` → `/train_model` → `/test_model` →
   `/predict` → `/reset`, confirming each JSON response's top-level keys exactly
   match what `templates/index.html`'s JS reads (see field list established
   during CMAPSS-1's own verification) - `success`, `train_shape[0..1]`,
   `features_after_pca`, `test_shape[0]` / `final_loss`, `final_accuracy`, `plot`
   / `test_loss`, `test_accuracy`, `plot` / `prediction`, `status`, `plot`.
8. Exercise the actual UI in a browser through all 4 steps + reset, confirming
   the new Test Model step's button, loading spinner, and results render
   correctly, and that Previous/Next navigation works across all 4 steps.
