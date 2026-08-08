"""
GraphManager: pure rendering. No dataset-specific knowledge -- every method
takes plain arrays/scalars as arguments and returns a base64 PNG string.
Only imports PlotConfig; never DataConfig or ModelConfig.
"""
import io
import base64
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import PlotConfig


class GraphManager:

    @staticmethod
    def _fig_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_url

    @staticmethod
    def render_training_plot(history, metric_key='accuracy'):
        """Loss + accuracy curves, TRAINING ONLY (no val_* -- testing is a
        separate step now, so validation_data is never passed to fit())."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PlotConfig.TRAIN_FIGSIZE)

        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(history.history[metric_key], label='Training Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        return GraphManager._fig_to_base64(fig)

    @staticmethod
    def render_test_plot(y_true, y_pred_prob):
        """A post-hoc evaluate() call has no epoch history to plot -- this
        shows the predicted-probability distribution on the test set, split
        by true label, which is the only artifact evaluate() produces."""
        y_true = np.asarray(y_true)
        y_pred_prob = np.asarray(y_pred_prob)

        fig, ax = plt.subplots(figsize=PlotConfig.TEST_FIGSIZE)
        ax.hist(y_pred_prob[y_true == 0], bins=30, alpha=0.6,
                label='True: Normal (0)', color='steelblue')
        ax.hist(y_pred_prob[y_true == 1], bins=30, alpha=0.6,
                label='True: At Risk (1)', color='crimson')
        ax.axvline(x=PlotConfig.PROBABILITY_THRESHOLD, color=PlotConfig.THRESHOLD_LINE_COLOR,
                   linestyle=':', linewidth=2,
                   label=f'Threshold ({PlotConfig.PROBABILITY_THRESHOLD})')
        ax.set_title('Test Set: Predicted Probability Distribution by True Label')
        ax.set_xlabel('Predicted Failure Probability')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return GraphManager._fig_to_base64(fig)

    @staticmethod
    def render_prediction_plot(X_pred_scaled, feature_columns, predictions,
                                failure_idx, time_steps, predict_unit):
        """Top: selected sensor curves + failure-cycle vline.
        Bottom: probability-over-time + threshold hline + failure-cycle vline."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PlotConfig.PREDICT_FIGSIZE)

        time_indices = range(len(X_pred_scaled))
        n_features = X_pred_scaled.shape[1]
        for i in range(n_features):
            ax1.plot(time_indices, X_pred_scaled[:, i], label=feature_columns[i], alpha=0.7)
        ax1.axvline(x=failure_idx, color=PlotConfig.FAILURE_LINE_COLOR, linestyle=':', label='Failure cycle')
        ax1.set_title(f'Selected Sensor Features Over Time ({n_features} sensors) -- unit {predict_unit}')
        ax1.set_xlabel('Cycle')
        ax1.set_ylabel('Feature Value')
        # Legend placed outside the axes (to the right) so it never overlaps
        # the plotted curves -- with up to 10 sensors + the failure line,
        # an in-plot legend was obscuring data near the end of the series.
        # _fig_to_base64's bbox_inches='tight' expands the saved image to
        # include it, so nothing gets clipped.
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        ax1.grid(True)

        pred_time_indices = range(time_steps, time_steps + len(predictions))
        ax2.plot(pred_time_indices, predictions, color=PlotConfig.PREDICTION_LINE_COLOR, linewidth=2,
                 label='Predicted Failure Probability')
        ax2.axhline(y=PlotConfig.PROBABILITY_THRESHOLD, color=PlotConfig.THRESHOLD_LINE_COLOR,
                   linestyle=':', alpha=0.7, label=f'Threshold ({PlotConfig.PROBABILITY_THRESHOLD})')
        ax2.axvline(x=failure_idx, color=PlotConfig.FAILURE_LINE_COLOR, linestyle=':', label='Failure cycle')
        ax2.set_title(f'Failure Probability Over Time (final = {predictions[-1]:.3f})')
        ax2.set_xlabel('Cycle')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return GraphManager._fig_to_base64(fig)
