let currentStep = 1;
let dataReady = false;
let modelTrained = false;
let modelTested = false;

function showAlert(message, type = 'info') {
    const alerts = document.getElementById('alerts');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    alerts.appendChild(alert);

    setTimeout(() => {
        alert.remove();
    }, 5000);
}

function showLoading(stepNum) {
    document.getElementById(`loading${stepNum}`).style.display = 'block';
}

function hideLoading(stepNum) {
    document.getElementById(`loading${stepNum}`).style.display = 'none';
}

function updateStepStatus(stepNum, status) {
    const step = document.getElementById(`step${stepNum}`);
    step.classList.remove('active', 'completed');
    step.classList.add(status);
}

// Previous button reads as a plain neutral gray (.btn-secondary)
// until a prediction has actually been made, then switches to the
// same active blue every other action button uses -- a visual cue
// that the wizard's work is done and it's fine to navigate back.
function setPrevButtonActive(isActive) {
    const prevBtn = document.getElementById('prevBtn');
    if (isActive) {
        prevBtn.classList.remove('btn-secondary');
    } else {
        prevBtn.classList.add('btn-secondary');
    }
}

function showStep(stepNum) {
    // Hide all content
    document.querySelectorAll('.step-content').forEach(content => {
        content.classList.remove('active');
    });

    // Show current content
    document.getElementById(`content${stepNum}`).classList.add('active');

    // Update step indicators
    document.querySelectorAll('.step').forEach((step, index) => {
        step.classList.remove('active');
        if (index + 1 === stepNum) {
            step.classList.add('active');
        }
    });

    // Update navigation buttons
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    prevBtn.style.display = stepNum > 1 ? 'block' : 'none';

    if (stepNum === 1) {
        nextBtn.style.display = dataReady ? 'block' : 'none';
    } else if (stepNum === 2) {
        nextBtn.style.display = modelTrained ? 'block' : 'none';
    } else if (stepNum === 3) {
        nextBtn.style.display = modelTested ? 'block' : 'none';
    } else {
        nextBtn.style.display = 'none';
    }
}

function nextStep() {
    if (currentStep < 4) {
        currentStep++;
        showStep(currentStep);
    }
}

function previousStep() {
    if (currentStep > 1) {
        currentStep--;
        showStep(currentStep);
    }
}

function prepareData() {
    showLoading(1);

    fetch('/prepare_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            train_files: [],
            test_files: []
        })
    })
    .then(response => response.json())
    .then(result => {
        hideLoading(1);

        if (result.success) {
            showAlert('Data preparation completed successfully!', 'success');
            updateStepStatus(1, 'completed');
            dataReady = true;

            // Show results
            const resultsDiv = document.createElement('div');
            resultsDiv.className = 'metrics';
            resultsDiv.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${result.train_shape[0]}</div>
                    <div class="metric-label">Training Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${result.train_shape[1]}</div>
                    <div class="metric-label">Time Steps</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${result.features_after_pca}</div>
                    <div class="metric-label">Selected Sensors</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${result.test_shape[0]}</div>
                    <div class="metric-label">Test Samples</div>
                </div>
            `;

            document.getElementById('content1').appendChild(resultsDiv);
            showStep(1);
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            showAlert('Error preparing data: ' + result.error, 'error');
        }
    })
    .catch(error => {
        hideLoading(1);
        showAlert('Error preparing data: ' + error.message, 'error');
    });
}

function trainModel() {
    showLoading(2);
    document.getElementById('training-results').innerHTML = '';

    const epochs = parseInt(document.getElementById('epochs').value);
    const learningRate = parseFloat(document.getElementById('learning_rate').value);

    fetch('/train_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            epochs: epochs,
            learning_rate: learningRate
        })
    })
    .then(response => response.json())
    .then(result => {
        hideLoading(2);

        if (result.success) {
            showAlert('Model training completed successfully!', 'success');
            updateStepStatus(2, 'completed');
            modelTrained = true;

            // Retraining invalidates any test/prediction results
            // shown for the PREVIOUS model -- clear them and revert
            // their step indicators so stale results/graphs from an
            // old model can't be mistaken for current ones.
            modelTested = false;
            document.getElementById('test-results').innerHTML = '';
            document.getElementById('prediction-results').innerHTML = '';
            document.getElementById('step3').classList.remove('completed');
            document.getElementById('step4').classList.remove('completed');
            setPrevButtonActive(false);

            // Show results
            const resultsDiv = document.getElementById('training-results');
            resultsDiv.innerHTML = `
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">${result.final_loss.toFixed(4)}</div>
                        <div class="metric-label">Final Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(result.final_accuracy * 100).toFixed(1)}%</div>
                        <div class="metric-label">Final Accuracy</div>
                    </div>
                </div>
                <div class="plot-container">
                    <h3>Training Progress</h3>
                    <img src="data:image/png;base64,${result.plot}" alt="Training Plot">
                </div>
            `;

            showStep(2);
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            showAlert('Error training model: ' + result.error, 'error');
        }
    })
    .catch(error => {
        hideLoading(2);
        showAlert('Error training model: ' + error.message, 'error');
    });
}

function testModel() {
    showLoading(3);
    document.getElementById('test-results').innerHTML = '';

    fetch('/test_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(result => {
        hideLoading(3);

        if (result.success) {
            showAlert('Model testing completed successfully!', 'success');
            updateStepStatus(3, 'completed');
            modelTested = true;

            // Show results
            const resultsDiv = document.getElementById('test-results');
            resultsDiv.innerHTML = `
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">${result.test_loss.toFixed(4)}</div>
                        <div class="metric-label">Test Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(result.test_accuracy * 100).toFixed(1)}%</div>
                        <div class="metric-label">Test Accuracy</div>
                    </div>
                </div>
                <div class="plot-container">
                    <h3>Test Results</h3>
                    <img src="data:image/png;base64,${result.plot}" alt="Test Plot">
                </div>
            `;

            showStep(3);
        } else {
            showAlert('Error testing model: ' + result.error, 'error');
        }
    })
    .catch(error => {
        hideLoading(3);
        showAlert('Error testing model: ' + error.message, 'error');
    });
}

function skipTest() {
    // Testing is diagnostic, not a prerequisite for prediction --
    // this bypasses the /test_model call entirely and goes straight
    // to the Prediction step without marking Test as completed.
    currentStep = 4;
    showStep(4);
}

function getSelectedUnit() {
    const val = document.getElementById('predict-unit').value;
    return val ? parseInt(val) : null;
}

function selectMode(mode) {
    const staticPanel = document.getElementById('static-panel');
    const dynamicPanel = document.getElementById('dynamic-panel');

    if (mode === 'dynamic') {
        document.getElementById('mode-dynamic-radio').checked = true;
        staticPanel.style.display = 'none';
        dynamicPanel.style.display = 'block';
    } else {
        document.getElementById('mode-static-radio').checked = true;
        dynamicPanel.style.display = 'none';
        staticPanel.style.display = 'block';
        stopRealtimePrediction();   // leaving dynamic mode closes any open stream
    }
}

function selectTrainingMode(mode) {
    const syncPanel = document.getElementById('sync-train-panel');
    const livePanel = document.getElementById('live-train-panel');

    if (mode === 'live') {
        document.getElementById('mode-live-radio').checked = true;
        syncPanel.style.display = 'none';
        livePanel.style.display = 'block';
    } else {
        document.getElementById('mode-sync-radio').checked = true;
        livePanel.style.display = 'none';
        syncPanel.style.display = 'block';
        stopLiveTraining();   // leaving live mode closes any open stream
    }
}

function makePrediction() {
    const unit = getSelectedUnit();
    if (!unit) {
        showAlert('Please select an engine unit before predicting.', 'error');
        return;
    }

    showLoading(4);
    document.getElementById('prediction-results').innerHTML = '';

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ unit: unit })
    })
    .then(response => response.json())
    .then(result => {
        hideLoading(4);

        if (result.success) {
            showAlert('Prediction completed successfully!', 'success');
            updateStepStatus(4, 'completed');
            setPrevButtonActive(true);

            // Show results
            const resultsDiv = document.getElementById('prediction-results');
            const predictionClass = result.prediction > 0.5 ? 'prediction-risk' : 'prediction-normal';

            resultsDiv.innerHTML = `
                <div class="prediction-result">
                    <h3>Prediction Result</h3>
                    <div class="prediction-value ${predictionClass}">
                        ${(result.prediction * 100).toFixed(1)}%
                    </div>
                    <p><strong>Status:</strong> ${result.status}</p>
                    <p><strong>Threshold:</strong> 50% (values above indicate failure risk)</p>
                </div>
                <div class="plot-container">
                    <h3>Sensor Data Analysis & Prediction</h3>
                    <img src="data:image/png;base64,${result.plot}" alt="Prediction Plot">
                </div>
            `;
        } else {
            showAlert('Error making prediction: ' + result.error, 'error');
        }
    })
    .catch(error => {
        hideLoading(4);
        showAlert('Error making prediction: ' + error.message, 'error');
    });
}

function resetApplication() {
    if (confirm('Are you sure you want to reset the application? This will clear all data and models.')) {
        fetch('/reset')
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                showAlert('Application reset successfully!', 'success');

                // Reset UI state
                currentStep = 1;
                dataReady = false;
                modelTrained = false;
                modelTested = false;

                // Clear results
                document.getElementById('training-results').innerHTML = '';
                document.getElementById('test-results').innerHTML = '';
                document.getElementById('prediction-results').innerHTML = '';
                stopRealtimePrediction();
                document.getElementById('realtime-status').innerHTML = '';
                selectMode('static');
                stopLiveTraining();
                document.getElementById('live-train-status').innerHTML = '';
                selectTrainingMode('sync');
                setPrevButtonActive(false);

                // Reset step indicators
                document.querySelectorAll('.step').forEach((step, index) => {
                    step.classList.remove('active', 'completed');
                    if (index === 0) {
                        step.classList.add('active');
                    }
                });

                // Clear any additional content
                const content1 = document.getElementById('content1');
                const metrics = content1.querySelector('.metrics');
                if (metrics) {
                    metrics.remove();
                }

                showStep(1);
            }
        })
        .catch(error => {
            showAlert('Error resetting application: ' + error.message, 'error');
        });
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('reset-app-btn').addEventListener('click', resetApplication);
    document.getElementById('prepare-data-btn').addEventListener('click', prepareData);
    document.getElementById('train-model-btn').addEventListener('click', trainModel);
    document.getElementById('test-model-btn').addEventListener('click', testModel);
    document.getElementById('skip-test-btn').addEventListener('click', skipTest);
    document.getElementById('make-prediction-btn').addEventListener('click', makePrediction);
    document.getElementById('prevBtn').addEventListener('click', previousStep);
    document.getElementById('nextBtn').addEventListener('click', nextStep);

    document.getElementById('mode-sync-radio').addEventListener('change', () => selectTrainingMode('sync'));
    document.getElementById('mode-live-radio').addEventListener('change', () => selectTrainingMode('live'));
    document.getElementById('mode-static-radio').addEventListener('change', () => selectMode('static'));
    document.getElementById('mode-dynamic-radio').addEventListener('change', () => selectMode('dynamic'));

    showStep(1);
});
