"""
### `visualise.py`
- Provide reusable plotting functions using `matplotlib`
- Must be usable across scripts, demos, or pipeline logs
- Required plots:
- `plot_metric_over_time(metric_values, title, ylabel)`: Plot a metric (e.g.,
accuracy) across chunks
- `compare_models(metric1, metric2, labels)`: Compare two models on streaming
metrics (e.g., trees with different `buffer_size` values)
- `plot_predictions_vs_ground_truth(y_true, y_pred)`: Visualise predictions vs.
actuals on latest chunk
- Should support options for saving to file or inline display 
"""