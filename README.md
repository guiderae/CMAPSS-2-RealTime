# CMAPSS-2: Turbofan Failure Prediction

An LSTM-based predictive-maintenance web app for NASA's C-MAPSS FD001 turbofan engine
dataset. Trains a failure-risk model on 100 simulated engine units and lets you run
predictions against held-out units two ways: **Static** (one-shot, against a unit's full
history) or **Dynamic** (a live, cycle-by-cycle simulated stream, updating a chart in
real time).

Built as an object-oriented Flask app -  see [`STAGE2_PLAN.md`](STAGE2_PLAN.md) for the
architecture (5 Manager classes, no global state, config-driven constants) and
[`PHASE3_SUMMARY.md`](PHASE3_SUMMARY.md) for how the real-time streaming mode works.

## Prerequisites

- Python 3.11
- ~200MB free disk space (TensorFlow + dependencies)

## Setup

```bash
git clone https://github.com/guiderae/CMAPSS-2-RealTime.git
cd CMAPSS-2-RealTime

python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The dataset (`data/train_FD001.txt`, NASA C-MAPSS FD001 - 100 turbofan units, each run to
failure) is already included in the repo; no separate download needed.

## Running

```bash
venv/bin/python app.py
```

The server starts on **http://localhost:8082**. Open that URL in a browser.

## Usage

The app is a 4-step wizard:

1. **Data Preparation** - click **Prepare Data**. Loads all 100 engine units, selects the
   10 sensors most correlated with approaching failure, and builds training/test sequences.
   Units 1–70 are used for training, 71–90 for testing, and 91 is held out entirely.

2. **Model Training** - set epochs/learning rate and click **Train Model**. Trains the LSTM
   from scratch each run; loss/accuracy curves are shown on completion.

3. **Test Model** - click **Test Model** to evaluate against the held-out test units, or
   **Skip Test & Proceed to Prediction** to go straight to predicting (testing is
   diagnostic, not required).

4. **Prediction** - pick an engine unit from the dropdown (units 71–91, all held out from
   training), then choose a mode:
   - **Static Prediction** - runs the model once against the unit's full history and shows
     the failure probability plus a plot.
   - **Dynamic Prediction** - click **Start** to stream the unit's sensor readings cycle by
     cycle (simulating live data arrival) into a chart with sensors on the left axis and a
     live failure-probability trace on the right axis. The probability trace stays empty
     for the first 30 cycles (the model needs that much history before it can predict
     anything), then begins tracking. **Stop** ends the stream early.

**Reset Application** (top right) clears all state - data, trained model, and any
results/charts on screen - back to step 1.

## Project structure

```
app.py                    Flask routes (thin controller - no business logic)
config.py                 All constants, grouped by concern
lstm_model.py             LSTM architecture + the shared trained model instance
managers/                 Business logic: data prep, training, testing, static prediction, plotting
realtime/                 Dynamic-prediction streaming: paced data generator + SSE message builder
utils/                    Helper for populating the unit-selector dropdown
templates/index.html      The single-page wizard UI
static/js/realtime.js     Browser-side EventSource + Plotly chart logic for Dynamic Prediction
data/train_FD001.txt      NASA C-MAPSS FD001 dataset (100 engine units)
```

## Notes

- Training is not seeded to a fixed random state across processes in a way that guarantees
  bit-identical results between runs on different machines, but results within a run are
  reproducible (see `config.py`'s `ModelConfig.RANDOM_SEED`).
- The Flask dev server runs with `threaded=True` - required so a live Dynamic Prediction
  stream doesn't block other requests (e.g. Reset) while it's running.
