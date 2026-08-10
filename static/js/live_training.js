// Wires the Live Training Start/Stop buttons to a Server-Sent Events stream
// from /train_model_realtime, and renders one dual-axis Plotly chart: loss
// on the left y-axis, accuracy on the right y-axis (0-1), a continuously
// incrementing per-batch step count as x-axis.

let liveTrainingSource = null;
const LIVE_TRAIN_CHART_DIV = 'live-train-chart';

function startLiveTraining() {
    if (liveTrainingSource) {
        stopLiveTraining();
    }

    const epochs = parseInt(document.getElementById('epochs').value);
    const learningRate = parseFloat(document.getElementById('learning_rate').value);

    document.getElementById('start-live-train-btn').disabled = true;
    document.getElementById('stop-live-train-btn').disabled = false;
    document.getElementById('live-train-status').innerHTML = '';

    // The chart lives below the fold at this point in the page -- bring
    // it into view right away so the user isn't left watching a static
    // screen while training is actually running further down.
    document.getElementById(LIVE_TRAIN_CHART_DIV).scrollIntoView({ behavior: 'smooth', block: 'start' });

    const url = `/train_model_realtime?epochs=${encodeURIComponent(epochs)}&learning_rate=${encodeURIComponent(learningRate)}`;
    liveTrainingSource = new EventSource(url);

    liveTrainingSource.addEventListener('initialize', function (event) {
        const payload = JSON.parse(event.data);
        // Unlike Dynamic Prediction's rolling window (old readings scroll
        // off, only recent behavior matters), a training run's whole
        // shape -- where loss drops fastest, whether it plateaus -- only
        // reads correctly with every batch visible at once. So the x-axis
        // is fixed to the full run up front and no points are ever
        // trimmed (see updateLiveTrainingChart).
        const totalSteps = payload.epochs * payload.steps_per_epoch;
        initLiveTrainingChart(totalSteps, payload.epochs, payload.steps_per_epoch);
        document.getElementById('live-train-status').innerHTML =
            `<p>Training started &mdash; ${payload.epochs} epochs, ${payload.steps_per_epoch} batches/epoch.</p>`;
    });

    liveTrainingSource.addEventListener('update', function (event) {
        updateLiveTrainingChart(JSON.parse(event.data));
    });

    liveTrainingSource.addEventListener('error', function (event) {
        // Custom server-sent "error" event (e.g. training already in
        // progress, or an exception inside the background fit() thread).
        // MUST close() here, or EventSource's default auto-reconnect
        // would silently retry and potentially start a fresh run once
        // whatever's holding the lock finishes.
        let message = 'Live training stream encountered an error.';
        try {
            const payload = JSON.parse(event.data);
            if (payload && payload.error) message = payload.error;
        } catch (e) { /* malformed error payload; use default message */ }
        showAlert(message, 'error');
        stopLiveTraining();
    });

    liveTrainingSource.addEventListener('jobfinished', function () {
        // Close BEFORE the async POST -- same auto-reconnect hazard as
        // the error handler above; the stream ending on its own (server
        // generator returns) is indistinguishable from a dropped
        // connection to EventSource unless we close() explicitly here.
        stopLiveTraining();
        fetch('/train_model_realtime_complete', { method: 'POST' })
            .then(response => response.json())
            .then(result => {
                if (!result.success) {
                    showAlert('Live training finished but failed to confirm completion.', 'error');
                    return;
                }
                showAlert('Live training completed successfully!', 'success');
                updateStepStatus(2, 'completed');
                modelTrained = true;

                // Mirror trainModel()'s success-path cleanup exactly:
                // retraining invalidates any test/prediction results from
                // the previous model.
                modelTested = false;
                document.getElementById('test-results').innerHTML = '';
                document.getElementById('prediction-results').innerHTML = '';
                document.getElementById('step3').classList.remove('completed');
                document.getElementById('step4').classList.remove('completed');
                setPrevButtonActive(false);

                showStep(2);
            })
            .catch(error => {
                showAlert('Error confirming live training completion: ' + error.message, 'error');
            });
    });

    liveTrainingSource.onerror = function () {
        // EventSource's own built-in connection-level error (e.g. dropped/
        // refused connection) -- distinct from the custom "error" event
        // above. Also must close() to prevent auto-reconnect.
        showAlert('Live training stream connection was lost.', 'error');
        stopLiveTraining();
    };
}

function stopLiveTraining() {
    if (liveTrainingSource) {
        liveTrainingSource.close();
        liveTrainingSource = null;
    }
    document.getElementById('start-live-train-btn').disabled = false;
    document.getElementById('stop-live-train-btn').disabled = true;
}

function initLiveTrainingChart(totalSteps, epochs, stepsPerEpoch) {
    const traces = [
        { x: [], y: [], mode: 'lines', name: 'Loss', yaxis: 'y1', line: { width: 2, color: 'crimson' } },
        { x: [], y: [], mode: 'lines', name: 'Accuracy', yaxis: 'y2', line: { width: 2, color: 'steelblue' } }
    ];

    const layout = {
        title: 'Live Training Progress',
        xaxis: { title: 'Batch (cumulative)', type: 'linear', range: [1, totalSteps] },
        yaxis: { title: 'Loss', side: 'left', range: [0, 1] },
        yaxis2: { title: 'Accuracy', side: 'right', overlaying: 'y', range: [0, 1] },
        legend: { orientation: 'h' },
        margin: { t: 120 },   // extra headroom above the plot for the epoch labels below
        shapes: buildEpochBoundaryShapes(epochs, stepsPerEpoch),
        annotations: buildEpochLabels(epochs, stepsPerEpoch)
    };

    Plotly.newPlot(LIVE_TRAIN_CHART_DIV, traces, layout);
}

function buildEpochLabels(epochs, stepsPerEpoch) {
    // "Epoch N" centered above each epoch's span of batches, sitting in
    // the margin reserved above the plot area (yref: 'paper', y > 1).
    // With a lot of epochs the labels would overlap, so thin them out to
    // roughly one per 40px-ish of width -- in practice, at most ~20 labels
    // shown, spaced evenly across however many epochs there actually are.
    const maxLabels = 20;
    const labelEvery = Math.max(1, Math.ceil(epochs / maxLabels));

    const annotations = [];
    for (let epoch = 0; epoch < epochs; epoch += labelEvery) {
        const xMid = epoch * stepsPerEpoch + (stepsPerEpoch / 2) + 0.5;
        annotations.push({
            x: xMid,
            y: 1.06,
            xref: 'x',
            yref: 'paper',
            text: 'Epoch ' + (epoch + 1),
            showarrow: false,
            font: { size: 10, color: '#666' }
        });
    }
    return annotations;
}

function buildEpochBoundaryShapes(epochs, stepsPerEpoch) {
    // One dotted vertical line between each pair of consecutive epochs --
    // e.g. for epochs=20 that's 19 lines, none at the very start or end.
    // The +0.5 offset sits the line between the last batch of one epoch
    // and the first of the next, rather than directly on top of a point.
    // yref: 'paper' spans the full plot height regardless of the loss
    // y-axis's own [0,1] range, so the line always reaches top to bottom.
    const shapes = [];
    for (let epoch = 1; epoch < epochs; epoch++) {
        const x = epoch * stepsPerEpoch + 0.5;
        shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: x,
            x1: x,
            y0: 0,
            y1: 1,
            line: { color: 'gray', width: 1, dash: 'dot' }
        });
    }
    return shapes;
}

function updateLiveTrainingChart(payload) {
    // No maxPoints arg here -- unlike Dynamic Prediction's use of
    // MAX_CHART_POINTS, every batch must stay in the trace for the full
    // run's shape to be visible (see the 'initialize' handler above).
    Plotly.extendTraces(
        LIVE_TRAIN_CHART_DIV,
        { x: [[payload.step], [payload.step]], y: [[payload.loss], [payload.accuracy]] },
        [0, 1]
    );

    document.getElementById('live-train-status').innerHTML =
        `<p>Epoch ${payload.epoch + 1}, batch ${payload.batch + 1} &mdash; ` +
        `loss: <strong>${payload.loss.toFixed(4)}</strong>, ` +
        `accuracy: <strong>${(payload.accuracy * 100).toFixed(1)}%</strong></p>`;
}
