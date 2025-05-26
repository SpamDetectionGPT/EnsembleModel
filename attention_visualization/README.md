# Transformer Attention Visualization Tools

This package provides tools for extracting and visualizing attention patterns from transformer models. It's specifically designed to work with the encoder-only transformer model trained for sequence classification, allowing you to extract the attention matrices and visualize how the model attends to different tokens in the input.

## Contents

- `extract_attention.py`: Script to extract attention matrices from a trained transformer model
- `visualize_attention.py`: Script to visualize or prepare attention data for backend visualization
- `README.md`: This documentation file

## Installation

### Prerequisites

The tools require the following Python packages:

```bash
pip install torch numpy matplotlib transformers
```

### Configuration

No additional configuration is needed. The tools are designed to work with the model structure in this repository.

## Usage

### Extracting Attention Matrices

The `extract_attention.py` script loads your trained model and extracts attention matrices for a given input text:

```bash
python attention_visualization/extract_attention.py \
  --model_dir "./model_output" \
  --text "Your input text here" \
  --output_file "attention_data.json"
```

#### Parameters:

- `--model_dir`: Directory containing your trained model (default: "./model_output")
- `--text`: Input text to extract attention for (required)
- `--output_file`: File to save attention data to (default: "attention_data.json")
- `--device`: Device to use for inference (cuda, mps, cpu)
- `--layer`: Specific layer to extract attention from (default: all layers)

### Visualizing Attention

The `visualize_attention.py` script provides two modes of operation:

#### 1. Generating a visualization image:

```bash
python attention_visualization/visualize_attention.py \
  --input_file "attention_data.json" \
  --mode plot \
  --layer 0 \
  --head 0 \
  --output_file "attention_vis.png"
```

This generates a heatmap visualization of the attention matrix for a specific layer and attention head.

#### 2. Preparing data for backend visualization:

```bash
python attention_visualization/visualize_attention.py \
  --input_file "attention_data.json" \
  --mode prepare \
  --output_file "backend_data.json"
```

This processes the attention data into a format suitable for visualization in a web or other custom frontend.

#### Parameters:

- `--input_file`: JSON file containing attention data (required)
- `--output_file`: Output file for visualization or processed data
- `--mode`: Mode: "plot" (visualize) or "prepare" (for backend) (default: "plot")
- `--layer`: Layer index to visualize in plot mode (default: 0)
- `--head`: Head index to visualize in plot mode (default: 0)

## Data Format

### Extracted Attention Data

The output from `extract_attention.py` is a JSON file with the following structure:

```json
{
  "tokens": ["[CLS]", "token1", "token2", ..., "[PAD]", ...],
  "text": "original input text",
  "attention_matrices": [
    [
      [[attention_values_head_0_layer_0]], 
      [[attention_values_head_1_layer_0]], 
      ...
    ],
    [
      [[attention_values_head_0_layer_1]],
      ...
    ],
    ...
  ]
}
```

### Backend Visualization Data

The output from `visualize_attention.py` in prepare mode is a JSON file with the following structure:

```json
{
  "tokens": ["[CLS]", "token1", "token2", ...],
  "text": "original input text",
  "layers": [
    {
      "layer_idx": 0,
      "heads": [
        {
          "head_idx": 0,
          "attention": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
        },
        ...
      ]
    },
    ...
  ]
}
```

This processed format removes padding tokens and restructures the data for easier consumption by visualization libraries.

## Technical Details

### Attention Extraction Method

The attention extraction works by performing a custom forward pass through the model's layers:

1. The input text is tokenized
2. The tokens are passed through the model's embedding layer
3. For each transformer layer:
   - Apply layer normalization
   - For each attention head, extract query (q), key (k), and value (v) projections
   - Compute attention scores: `scores = (q·k^T)/sqrt(d_k)`
   - Apply softmax to get attention weights
   - Store these weights as the attention matrix
   - Continue with the standard forward pass to get the input for the next layer

This approach allows us to extract the actual attention weights used by the model during inference, providing insight into which tokens the model is focusing on for its predictions.

### Visualization Technique

The visualization uses a heatmap to represent the attention matrix, with brighter colors indicating higher attention values. The x and y axes represent tokens in the sequence, and each cell shows how much the model attends from one token (y-axis) to another token (x-axis).

## Examples

### Example 1: Classification with Attention Visualization

To analyze how your model classifies a specific example:

```bash
# Extract attention data
python attention_visualization/extract_attention.py \
  --text "This is an example text to analyze" \
  --output_file "example_attention.json"

# Generate visualization for first layer, first head
python attention_visualization/visualize_attention.py \
  --input_file "example_attention.json" \
  --layer 0 \
  --head 0 \
  --output_file "example_layer0_head0.png"
```

### Example 2: Preparing Data for Custom Visualization

If you're building a custom visualization in a web application:

```bash
# Extract attention data
python attention_visualization/extract_attention.py \
  --text "This is an example text to analyze" \
  --output_file "example_attention.json"

# Prepare for backend
python attention_visualization/visualize_attention.py \
  --input_file "example_attention.json" \
  --mode prepare \
  --output_file "for_backend.json"
```

Then use `for_backend.json` in your web application to create interactive visualizations.

## Extending the Tools

### Adding Support for Other Model Architectures

The current implementation is designed for the encoder-only transformer in this repository. To support other architectures:

1. Modify the `_extract_attention` method in `AttentionExtractor` class to match the architecture of your model
2. Update the model loading code in `_load_model` if necessary

### Creating Custom Visualizations

The `prepare_for_backend` function in `visualize_attention.py` creates a clean data format that can be easily consumed by various visualization libraries. This data can be used with:

- D3.js for web-based interactive visualizations
- React/Vue.js components for web applications
- Python visualization libraries like Plotly for interactive plots
- Custom dashboard tools

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Ensure your Python environment has all required packages installed.
2. **CUDA out of memory**: Try using a smaller batch size or move to CPU if your GPU doesn't have enough memory.
3. **Tokenization errors**: Verify your model's tokenizer is compatible with the input text format.

### Contact & Support

For questions and support, please open an issue in the repository or contact the repository maintainer. 