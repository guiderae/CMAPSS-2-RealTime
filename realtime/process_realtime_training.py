"""
ProcessRealtimeTraining: runs TrainManager-equivalent fit() logic in a
background thread (via LiveTrainingCallback wired to a queue), and yields
SSE-formatted messages as each batch's stats arrive in the main
(request-handling) thread. Guards against a second concurrent training
run via LSTMModel.training_lock, and against a client disconnect leaving
the background thread running unattended (sets a cancel flag the callback
checks every batch, then joins the thread before releasing the lock).

SSE message shapes:
  event: initialize
  data: {"epochs": <int>, "steps_per_epoch": <int>, "metric": "accuracy"}

  event: update
  data: {"step": <int>, "epoch": <int>, "batch": <int>, "loss": <float>, "accuracy": <float>}

  event: jobfinished
  data: none

  event: error
  data: {"error": <string>}
"""
import json
import math
import queue
import threading

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import ModelConfig
from lstm_model import LSTMModel
from managers.data_prep_manager import DataPrepManager
from realtime.live_training_callback import LiveTrainingCallback


class ProcessRealtimeTraining:

    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.update_queue = queue.Queue()
        self.cancel_requested = False

    def process_points(self):
        """Generator of SSE-formatted strings. epochs/learning_rate are
        plain values resolved by the caller before this generator or the
        Response is constructed -- same reasoning as ProcessRealtimeData:
        never touch Flask's request/session from inside a streamed
        generator body (Flask's session cookie is finalized before the
        response body is ever iterated, so mutating it here would be
        silently lost -- see /train_model_realtime_complete in app.py for
        where session['model_trained'] actually gets set)."""
        if not LSTMModel.training_lock.acquire(blocking=False):
            yield self.__sse('error', {'error': 'Training already in progress'})
            return

        thread = None
        try:
            X_train = DataPrepManager.X_train
            y_train = DataPrepManager.y_train
            steps_per_epoch = max(1, math.ceil(len(X_train) / ModelConfig.BATCH_SIZE))

            yield self.__sse('initialize', {
                'epochs': self.epochs,
                'steps_per_epoch': steps_per_epoch,
                'metric': ModelConfig.METRICS[0]
            })

            thread = threading.Thread(target=self.__run_training, args=(X_train, y_train, steps_per_epoch))
            thread.start()

            while True:
                update = self.update_queue.get()
                if update is None:
                    yield self.__sse('jobfinished', None)
                    break
                if isinstance(update, Exception):
                    yield self.__sse('error', {'error': str(update)})
                    break
                yield self.__sse('update', update)
        finally:
            # Fires on normal completion, on the break-on-error path, AND
            # on client disconnect (GeneratorExit) -- in every case, ask
            # the background thread to stop at its next batch boundary and
            # wait for it before releasing the lock, so a dropped
            # connection can't leave training running unattended or the
            # lock held indefinitely.
            self.cancel_requested = True
            if thread is not None:
                thread.join()
            LSTMModel.training_lock.release()

    def __run_training(self, X_train, y_train, steps_per_epoch):
        try:
            np.random.seed(ModelConfig.RANDOM_SEED)
            tf.random.set_seed(ModelConfig.RANDOM_SEED)

            input_shape = (X_train.shape[1], X_train.shape[2])
            LSTMModel.create(input_shape, learning_rate=self.learning_rate)

            classes = np.unique(y_train)
            weights = compute_class_weight(ModelConfig.CLASS_WEIGHT_STRATEGY, classes=classes, y=y_train)
            class_weight = dict(zip(classes.tolist(), weights))

            callback = LiveTrainingCallback(self.update_queue, steps_per_epoch, lambda: self.cancel_requested)
            LSTMModel.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=ModelConfig.BATCH_SIZE,
                class_weight=class_weight,
                callbacks=[callback],
                verbose=0
            )
            self.update_queue.put(None)          # sentinel: success
        except Exception as e:
            self.update_queue.put(e)             # sentinel: failure

    @staticmethod
    def __sse(event, payload):
        data = 'none' if payload is None else json.dumps(payload)
        return f"event: {event}\ndata: {data}\n\n"
