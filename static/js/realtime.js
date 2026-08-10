// Wires the Dynamic Prediction Start/Stop buttons to a Server-Sent Events
// stream from /predict_realtime, and renders one dual-axis Plotly chart:
// selected sensors on the left y-axis (scaled units), live failure
// probability on the right y-axis (0-1), cycle number on a linear x-axis.
// MAX_CHART_POINTS comes from an inline <script> global set by index.html
// (sourced from RealtimeConfig.MAX_CHART_POINTS) -- not hardcoded here.

let realtimeSource = null;
let realtimeSensorNames = [];
const REALTIME_CHART_DIV = 'realtime-chart';

function startRealtimePrediction() {
    const unit = document.getElementById('predict-unit').value;

    if (!unit) {
        showAlert('Please select an engine unit before starting a dynamic prediction.', 'error');
        return;
    }

    if (realtimeSource) {
        stopRealtimePrediction();
    }

    document.getElementById('start-stream-btn').disabled = true;
    document.getElementById('stop-stream-btn').disabled = false;
    document.getElementById('realtime-status').innerHTML = '';

    // The chart lives below the fold at this point in the page -- bring
    // it into view right away so the user isn't left watching a static
    // screen while the stream is actually running further down.
    document.getElementById(REALTIME_CHART_DIV).scrollIntoView({ behavior: 'smooth', block: 'start' });

    realtimeSource = new EventSource(`/predict_realtime?unit=${encodeURIComponent(unit)}`);

    realtimeSource.addEventListener('initialize', function (event) {
        const payload = JSON.parse(event.data);
        realtimeSensorNames = payload.sensors;
        document.getElementById('dynamic-time-steps-label').textContent = payload.time_steps;
        initRealtimeChart(realtimeSensorNames);
        document.getElementById('realtime-status').innerHTML =
            `<p>Streaming engine ${payload.unit} &mdash; waiting for ${payload.time_steps} cycles of history before predictions begin...</p>`;
    });

    realtimeSource.addEventListener('update', function (event) {
        updateRealtimeChart(JSON.parse(event.data));
    });

    realtimeSource.addEventListener('jobfinished', function () {
        showAlert('Dynamic prediction stream finished.', 'success');
        updateStepStatus(4, 'completed');
        stopRealtimePrediction();
    });

    realtimeSource.onerror = function () {
        showAlert('Dynamic prediction stream encountered an error.', 'error');
        stopRealtimePrediction();
    };
}

function stopRealtimePrediction() {
    if (realtimeSource) {
        realtimeSource.close();
        realtimeSource = null;
    }
    document.getElementById('start-stream-btn').disabled = false;
    document.getElementById('stop-stream-btn').disabled = true;
}

function initRealtimeChart(sensorNames) {
    const traces = sensorNames.map(name => ({
        x: [], y: [], mode: 'lines', name: name, yaxis: 'y1', line: { width: 1 }
    }));

    traces.push({
        x: [], y: [], mode: 'lines', name: 'Failure Prediction', yaxis: 'y2',
        line: { width: 3, color: 'red' }
    });

    const layout = {
        title: 'Real-Time Sensor Readings and Failure Prediction',
        xaxis: { title: 'Cycle', type: 'linear' },
        yaxis: { title: 'Scaled Sensors', side: 'left' },
        yaxis2: { title: 'Failure Prediction', side: 'right', overlaying: 'y', range: [0, 1] },
        legend: { orientation: 'h' }
    };

    Plotly.newPlot(REALTIME_CHART_DIV, traces, layout);
}

function updateRealtimeChart(payload) {
    const sensorTraceIndices = realtimeSensorNames.map((_, i) => i);
    const sensorX = sensorTraceIndices.map(() => [payload.cycle]);
    const sensorY = realtimeSensorNames.map(name => [payload.sensors[name]]);

    Plotly.extendTraces(REALTIME_CHART_DIV, { x: sensorX, y: sensorY }, sensorTraceIndices, MAX_CHART_POINTS);

    if (payload.prediction !== null && payload.prediction !== undefined) {
        const predictionTraceIndex = realtimeSensorNames.length;
        Plotly.extendTraces(
            REALTIME_CHART_DIV,
            { x: [[payload.cycle]], y: [[payload.prediction]] },
            [predictionTraceIndex],
            MAX_CHART_POINTS
        );
        document.getElementById('realtime-status').innerHTML =
            `<p>Cycle ${payload.cycle} &mdash; Failure probability: <strong>${(payload.prediction * 100).toFixed(1)}%</strong></p>`;
    } else {
        document.getElementById('realtime-status').innerHTML =
            `<p>Cycle ${payload.cycle} &mdash; buffering, no prediction yet</p>`;
    }
}
