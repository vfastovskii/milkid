# Gradient Tracking for Bag-Attention Networks

This module provides functionality to track and visualize gradients in Bag-Attention Networks (embedder → attention/aggregator → predictor) during training. It helps with model interpretability and debugging by monitoring gradient flow through the network.

## Features

- Track gradients at key points in the model:
  - Embedder input/output gradients to identify influential conformer features
  - Attention weights gradients to see which instances the model relies on
  - Aggregator output gradients to assess instance contribution to bag-level output
  - Predictor input/output gradients to evaluate sensitivity of final predictions

- Log gradient statistics to MLflow:
  - Gradient norms to detect vanishing or exploding gradients
  - Gradient mean, standard deviation, min, and max values
  - Gradient histograms for distribution visualization

## How It Works

The gradient tracking module registers hooks on model components to capture gradients during the backward pass. It calculates statistics and logs them to MLflow at the end of each training epoch.

### Implementation Details

1. **Hook Registration**: Hooks are registered on model components to capture gradients during the backward pass.
2. **Gradient Collection**: Gradients are collected and statistics are calculated.
3. **MLflow Logging**: Statistics are logged to MLflow at the end of each training epoch.

## Interpreting Gradient Visualizations

### Embedder Gradients

- **First Layer Weights**: Large gradients indicate features that strongly influence the model's predictions. These are the input features the model is learning to pay attention to.
- **Output Gradients**: Show how changes in embeddings affect the final prediction. High gradients indicate instances that contribute significantly to the bag representation.

### Attention Weights Gradients

- Large gradients in attention weights indicate instances that the model is actively learning to focus on or ignore.
- Consistent patterns across epochs suggest stable attention mechanisms.
- Highly variable gradients might indicate instability in the attention mechanism.

### Aggregator Output Gradients

- These gradients show how changes in the aggregated bag representation affect the final prediction.
- They help understand which aspects of the aggregated representation are most important for the prediction.

### Predictor Gradients

- **Input Gradients**: Show how sensitive the final prediction is to changes in the bag representation.
- **Last Layer Weights**: Large gradients indicate features that directly influence the final prediction.

## Detecting Training Issues

### Vanishing Gradients

- Look for gradient norms that are consistently very small (close to zero) across epochs.
- This indicates that certain parts of the model are not learning effectively.

### Exploding Gradients

- Look for gradient norms that are very large or increasing rapidly across epochs.
- This can lead to unstable training and poor convergence.

### Dead Layers

- Layers with consistently zero or near-zero gradients are not learning.
- Check the gradient histograms for layers that show minimal activity.

## Example Interpretation Workflow

1. **Start with Predictor Gradients**: Understand what the model is sensitive to at the output level.
2. **Examine Attention Weights Gradients**: Identify which instances the model is learning to focus on.
3. **Look at Embedder Gradients**: Determine which input features are most influential.
4. **Monitor Gradient Norms Over Time**: Ensure all parts of the model are learning and there are no vanishing/exploding gradients.

## Integration with MLflow

Gradient statistics and visualizations are automatically logged to MLflow during training. You can view them in the MLflow UI under the "Artifacts" tab in the "grad_histograms" directory.

The following metrics are logged:
- `grad/{component_name}/norm`: L2 norm of the gradient
- `grad/{component_name}/mean`: Mean value of the gradient
- `grad/{component_name}/std`: Standard deviation of the gradient
- `grad/{component_name}/min`: Minimum value of the gradient
- `grad/{component_name}/max`: Maximum value of the gradient

Where `{component_name}` is one of:
- `embedder_first_layer_weight`
- `embedder_output`
- `aggregator_attention_weights`
- `aggregator_output`
- `predictor_input`
- `predictor_last_layer_weight`