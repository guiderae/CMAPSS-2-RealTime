# Phase 3: Dynamic (Real-Time) Prediction — Summary

**Date:** 2026-08-08
**Scope:** New real-time streaming prediction mode added to CMAPSS-2's Prediction step,
via Server-Sent Events (SSE) and Plotly.js.

## What this adds

Prior to Phase 3, the Prediction step did everything in one batch: `PredictManager.predict()`
ran the trained model against a unit's entire history at once and returned a finished plot.
Phase 3 adds a second way to watch the same model work: sensor readings for a chosen engine
unit are streamed to the browser one cycle at a time (simulating real-time arrival), with the
failure-probability prediction updating incrementally as each new reading arrives.

The pattern (SSE + Plotly.js) follows a reference demo at
[guiderae/WorkingDemos-RealTimeGraph1](https://github.com/guiderae/WorkingDemos-RealTimeGraph1),
adapted to compute genuine incremental predictions from CMAPSS-2's actual trained model,
rather than that demo's raw unpredicted data plumbing.

This was purely additive — every existing route, manager, and config constant from Stage 2
stayed unchanged; Static Prediction's behavior is byte-for-byte identical to before.

## What the user sees

The Prediction step (step 4) now has:
- An **engine-unit dropdown**, always visible, listing the 21 held-out units the model never
  trained on (engines 71–91) — used by both prediction modes.
- A choice between **Static Prediction** (unchanged) and **Dynamic Prediction** (new).
- Dynamic Prediction has Start/Stop buttons and one combined chart: selected sensors on the
  left y-axis, live failure-probability trace on the right y-axis (0–1), cycle number on the
  x-axis. The probability trace stays absent for the first 30 cycles (not enough history for
  the model yet), then begins tracking.
- Selecting neither mode's action without picking a unit first shows a warning popup.

## Architecture

Five new files, following Stage 2's established conventions (thin `app.py`, business logic
in dedicated classes, all constants in `config.py`):

```
CMAPSS-2/
├── config.py                       (+RealtimeConfig, +FlaskConfig.THREADED)
├── managers/predict_manager.py     (+optional `unit` param, defaults preserved)
├── app.py                          (+/predict_realtime route, unit-aware /predict, /)
├── utils/
│   └── data_file_manager.py        DataFileManager — lists selectable units
├── realtime/
│   ├── data_source_manager.py      DataSourceManager — paced row generator
│   └── process_realtime_data.py    ProcessRealtimeData — buffers, predicts, formats SSE
├── static/js/realtime.js           EventSource wiring + Plotly chart
└── templates/index.html            unit dropdown, mode toggle, chart markup
```

- **`DataFileManager.get_selectable_units()`** — returns `TEST_UNIT_RANGE + [PREDICT_UNIT]`
  (71–91), derived from existing config, not a new hardcoded list.
- **`DataSourceManager.stream_unit_rows(unit)`** — a generator that reuses
  `DataPrepManager`'s already-fitted scaler/selected_sensors/train_feature_means (not raw CSV
  rows, unlike the reference demo), sleeping `RealtimeConfig.STREAM_DELAY_SECONDS` (0.1s)
  between rows so streamed values are on the exact same footing as training/testing data.
- **`ProcessRealtimeData`** — consumes that generator, maintains a rolling
  `DataConfig.TIME_STEPS`-row (30) buffer, and yields three SSE event types:
  `initialize` (sensor names + unit, sent once), `update` (per-cycle sensor values +
  prediction-or-null), `jobfinished` (stream complete).
- **`/predict_realtime`** — guarded on `session.get('model_trained')` exactly like `/predict`;
  resolves `unit` from the query string before constructing the generator/`Response`, so the
  generator body never touches Flask's `request`/`session` (avoids the classic "working
  outside of request context" failure mode for streamed generators).
- **`app.run(..., threaded=True)`** — required because a single-threaded dev server would
  block every other request (including Reset) for the whole 15–35s a stream stays open.

## The critical correctness fix

The naive design — append each new row to the 30-row buffer, *then* predict from it — would
have silently reintroduced the exact off-by-one bug fixed earlier in this project's history
(documented in `Claude2-1/INVESTIGATION_SUMMARY.md`): at a unit's actual failure row, the
buffer would include the failure row itself, and the model would be asked to predict one step
*beyond* the end of the data using a window that improperly includes the row it's supposed to
be predicting.

The fix: **predict from the buffer before appending the current row**, matching
`create_sequences()`'s training convention exactly (window `i..i+29` predicts row `i+30`, the
row *after* the window). This means the window used to predict any row — including the final,
actual failure row — is always the 30 rows strictly *before* it, never including it.

**This was verified concretely, not just reasoned about:** streaming unit 91 end-to-end, the
final SSE `update` event (cycle 135, the unit's real failure row) reported
`"prediction": 0.9984493255615234` — an exact match with `PredictManager.predict(unit=91)`'s
static-mode result for the same unit. Both code paths agree because both now compute the same
thing the same way.

## Post-implementation fix: alert popup visibility

During the walkthrough, the "no unit selected" warning was found to fire but not visibly
appear. Root cause: `#alerts` (the app's shared notification container, used by every route's
error handling since Stage 2, not just Phase 3) had no positioning CSS — it rendered in normal
document flow at the very bottom of the page, below the entire wizard. Alerts triggered near
the top of a step (like this one) landed off-screen and auto-removed after 5 seconds before
ever being scrolled into view.

Fixed by giving `#alerts` `position: fixed; top: 20px; right: 20px; z-index: 9999` plus a
box-shadow on `.alert` for visual separation — a floating top-right popup, visible regardless
of scroll position. This fixes the same latent issue for every alert in the app, not just this
one case.

## Verification performed

| check | result |
|---|---|
| `GET /` | 21 unit options (71–91) rendered; Plotly CDN + `realtime.js` tags present |
| `/prepare_data` → `/train_model` | unchanged shapes vs. Stage 2 baseline: `(12030,30,10)` / `(3650,30,10)` / 10 sensors |
| Static Prediction, `unit=91` vs. no unit | **identical** results — default parameter fully backward-compatible |
| `/predict_realtime` guards | 400 with no trained model; 400 with invalid unit |
| SSE stream, unit 91 (135 rows) | correct `initialize` payload; exactly 30 `null` predictions then real values from cycle 31 onward; 135 total updates; `jobfinished` at the end |
| Off-by-one correctness | final prediction (cycle 135, the real failure row) exactly matches Static Prediction's result for the same unit |
| Threading | concurrent `/predict` returned in 0.21s while a ~13s stream was actively running |
| Browser walkthrough | unit dropdown, mode toggle, live dual-axis chart, Start/Stop, Reset mid-stream, and the alert popup (post-fix) all confirmed working by the user |

## What's unchanged

Steps 1–3 (Data Preparation, Training, Testing) and all of Stage 2's architecture — the 5
Manager classes, `LSTMModel`, session/class-state separation, config-driven constants — are
untouched. Static Prediction's request/response contract is identical to before Phase 3.
