# Transformer Encoder from Scratch

This project contains a Python implementation of a Transformer Encoder, adapted from a Jupyter Notebook, focusing on building the core components like Multi-Head Attention and Feed-Forward layers from scratch using PyTorch.

## Project Structure

```
training/
├── pyproject.toml    # Project metadata and dependencies for uv/pip
├── models.py         # Contains PyTorch nn.Module classes for the Transformer components
├── main.py           # Simple script for running basic inference on a single sentence
├── train.py          # Main script for training the model on a dataset
├── data/
│   ├── __init__.py
│   └── preparation.py # Data loading and tokenization logic
├── utils/
│   ├── __init__.py
│   └── helpers.py    # Utility functions (device setup, argument parsing)
└── README.md         # This file
```

## Setup using uv

It is recommended to use a virtual environment.

1.  **Install uv (if you haven't already):**

    ```bash
    # Linux/macOS
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Alternatively, see the [official uv installation guide](https://github.com/astral-sh/uv#installation).

2.  **Create and activate a virtual environment:**
    Navigate to the `training` directory in your terminal.

    ```bash
    # Create a virtual environment named .venv
    uv venv

    # Activate the virtual environment
    # Linux/macOS
    source .venv/bin/activate
    # Windows (Command Prompt)
    .venv\Scripts\activate.bat
    # Windows (PowerShell)
    .venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    While inside the `training` directory with the virtual environment activated:
    ```bash
    uv pip install -e .
    ```
    This command installs the packages listed in `pyproject.toml` (torch, transformers, datasets, tqdm) and installs the current project (`training`) in editable mode.

## Running Basic Inference

Ensure your virtual environment is activated.

Navigate to the `training` directory and run the `main.py` script for a quick check on a single sentence:

```bash
python main.py
```

## Running Training

Ensure your virtual environment is activated.

Navigate to the `training` directory and run the `train.py` script.

**Basic Usage (using default IMDB dataset):**

```bash
python train.py
```

**Usage with Arguments:**

You can customize the training process using command-line arguments. Here are some examples:

```bash
# Train on a different dataset (e.g., 'ag_news') with 4 labels
python train.py --dataset_name ag_news --num_labels 4 --text_column text --label_column label

# Train for more epochs with a different learning rate and batch size
python train.py --epochs 5 --lr 1e-5 --batch_size 16

# Specify a different output directory for the saved model
python train.py --output_dir ./my_custom_model
```

Run `python train.py --help` to see all available arguments.

This will:

- Load the specified configuration and tokenizer (default: `bert-base-uncased`).
- Load and preprocess the specified dataset (default: `imdb`).
- Instantiate the custom `TransformerForSequenceClassification` model.
- Train the model for the specified number of epochs.
- Perform validation after each epoch (if a validation/test split is available).
- Save the trained model weights, configuration file, and tokenizer to the output directory (default: `./model_output`).
