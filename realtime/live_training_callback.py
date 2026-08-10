"""
LiveTrainingCallback: a Keras Callback that pushes per-batch training
stats onto a thread-safe queue -- the same producer role DataSourceManager
plays for Dynamic Prediction, just driven by Keras's own batch-end hook
instead of a paced file-row generator. Runs inside the background training
thread; the queue is drained by ProcessRealtimeTraining's SSE generator in
the main request thread.
"""
from tensorflow.keras.callbacks import Callback

from config import ModelConfig


class LiveTrainingCallback(Callback):

    def __init__(self, update_queue, steps_per_epoch, cancel_check):
        super().__init__()
        self.update_queue = update_queue
        self.steps_per_epoch = steps_per_epoch
        self.cancel_check = cancel_check   # zero-arg callable returning bool
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        if self.cancel_check():
            self.model.stop_training = True   # halts fit() before the next batch starts
            return
        logs = logs or {}
        self.step += 1
        # 0-indexed math throughout: at self.step == steps_per_epoch (the
        # LAST batch of epoch 0), this must still report epoch 0, not 1.
        epoch = (self.step - 1) // self.steps_per_epoch
        batch_in_epoch = (self.step - 1) % self.steps_per_epoch
        self.update_queue.put({
            'step': self.step,
            'epoch': epoch,
            'batch': batch_in_epoch,
            'loss': float(logs.get('loss', 0.0)),
            'accuracy': float(logs.get(ModelConfig.METRICS[0], 0.0))
        })
