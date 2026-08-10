from flask import Flask, render_template, request, jsonify, session, Response

from config import DataConfig, ModelConfig, FlaskConfig, RealtimeConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from managers.train_manager import TrainManager
from managers.test_manager import TestManager
from managers.predict_manager import PredictManager
from utils.data_file_manager import DataFileManager
from realtime.process_realtime_data import ProcessRealtimeData
from realtime.process_realtime_training import ProcessRealtimeTraining

app = Flask(__name__)
app.secret_key = FlaskConfig.SECRET_KEY


@app.route('/')
def index():
    all_units = DataPrepManager.get_unit_ids()
    train_units, test_units, predict_units = DataPrepManager.get_unit_splits(all_units)
    return render_template(
        'index.html',
        selectable_units=DataFileManager.get_selectable_units(),
        max_chart_points=RealtimeConfig.MAX_CHART_POINTS,
        total_units=len(all_units),
        train_units=train_units,
        test_units=test_units,
        predict_units=predict_units,
        time_steps=DataConfig.TIME_STEPS,
        top_n_sensors=DataConfig.TOP_N_SENSORS
    )


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


@app.route('/train_model_realtime')
def train_model_realtime_route():
    if not session.get('data_prepared'):
        return jsonify({'success': False, 'error': 'Data not prepared'}), 400
    epochs = request.args.get('epochs', ModelConfig.DEFAULT_EPOCHS, type=int)
    learning_rate = request.args.get('learning_rate', ModelConfig.DEFAULT_LEARNING_RATE, type=float)

    response = Response(ProcessRealtimeTraining(epochs, learning_rate).process_points(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/train_model_realtime_complete', methods=['POST'])
def train_model_realtime_complete_route():
    """Called by the client after it sees 'jobfinished' -- an ordinary
    request/response, so session mutation actually works (Flask's session
    cookie is finalized before an SSE generator's body ever runs, so
    /train_model_realtime itself can't set this)."""
    session['model_trained'] = True
    return jsonify({'success': True})


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
        body = request.json or {}
        unit = int(body['unit']) if body.get('unit') else None
        result = PredictManager.predict(unit=unit)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict_realtime')
def predict_realtime_route():
    if not session.get('model_trained'):
        return jsonify({'success': False, 'error': 'Model not trained'}), 400
    unit = request.args.get('unit', type=int)
    if unit is None or unit not in DataFileManager.get_selectable_units():
        return jsonify({'success': False, 'error': 'Invalid or missing unit'}), 400

    response = Response(ProcessRealtimeData(unit).process_points(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


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
    app.run(port=FlaskConfig.PORT, debug=FlaskConfig.DEBUG, threaded=FlaskConfig.THREADED)
