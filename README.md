# SpamGPT: Advanced Spam Detection Engine

SpamGPT is a sophisticated spam detection system featuring a custom decoder-only transformer model (SpamGPT) and an ensemble model that combines SpamGPT with a pre-trained encoder model. It provides comprehensive tools for training, evaluation, and inference.

## Overview

This project aims to accurately classify text messages (e.g., emails) as spam or ham (non-spam). It leverages a decoder-only transformer architecture, inspired by GPT models, tailored for this classification task. Additionally, an ensemble model is implemented to potentially improve performance by combining the strengths of the custom decoder and a general-purpose pre-trained encoder.

## Features

- **Custom Decoder-Only Transformer (SpamGPT)**:
    - Architecture similar to GPT, designed for sequence classification.
    - Handles special tokens for message boundaries (`<SOE>`, `<EOE>`), prediction prompts (`<SOP>`), and class labels (`<SPAM>`, `<HAM>`).
    - Causal masking in attention layers.
- **Ensemble Model**:
    - Combines predictions from SpamGPT (decoder) and a pre-trained transformer encoder (e.g., from Hugging Face).
    - Supports various ensembling methods like weighted averaging, voting, and a meta-learner.
- **Comprehensive Training Pipeline**:
    - Distributed Data Parallel (DDP) support for multi-GPU training.
    - Validation during training.
    - Checkpoint saving for best performing models.
    - Gradient accumulation and clipping.
    - Auxiliary loss for direct spam/ham classification during SpamGPT training.
- **Detailed Evaluation**:
    - Calculates Precision, Recall, F1-score, Accuracy, and Confusion Matrix.
    - Evaluates ensemble model and individual base models.
    - Saves detailed evaluation results in JSON format.
- **Inference Capabilities**:
    - Run inference with SpamGPT or the ensemble model.
    - Batch processing for multiple texts.
    - Attention visualization for SpamGPT.
- **Utilities**:
    - GPU monitoring script.
    - Wandb (Weights & Biases) integration for experiment tracking.
    - Discord webhook notifications for training/inference events.
    - Configurable model parameters.

## Model Architectures

### 1. SpamGPT (Decoder-Only)

- **Block Size**: Configurable (default: 1024 tokens, as seen in `config.py`)
- **Vocabulary Size**: 50,263 tokens (GPT-2 base + custom special tokens)
- **Layers**: Configurable (default: 12)
- **Attention Heads**: Configurable (default: 12)
- **Embedding Dimensions**: Configurable (default: 768)
- **Activation Function**: LeakyReLU in MLP layers (as per original README, may need verification in `spamGPT.py`)
- **Special Tokens**:
    - `<SOE>`: Start of Email/Message
    - `<EOE>`: End of Email/Message
    - `<SOP>`: Start of Prediction (model should predict `<HAM>` or `<SPAM>` after this)
    - `<EOP>`: End of Prediction
    - `<SPAM>`: Spam label token
    - `<HAM>`: Ham label token

### 2. Ensemble Model

- Combines a pre-trained encoder model (e.g., BERT, RoBERTa from Hugging Face, specified by `encoder_model_path`) and the trained SpamGPT model (decoder).
- **Methods**:
    - `weighted_average`: Combines probabilities from both models using specified weights.
    - `voting`: Uses majority vote from model predictions.
    - `meta_learner`: A small neural network trained to combine the outputs (probabilities or logits) of the base models.
- The `StackedEnsembleModel` class in `ensemble_model.py` and `ensemble_evaluation.py` implements this.

## File Structure

```
.
├── README.md                   # This file
├── requirements.txt            # Full list of Conda environment packages
├── pyproject.toml              # Project metadata and core dependencies
├── config.py                   # Configuration for SpamGPT model (block_size, vocab_size, etc.)
├── spamGPT.py                  # Implementation of the SpamGPT decoder-only model
├── dataset.py                  # Script for creating PyTorch DataLoaders
├── train.py                    # Script for training the SpamGPT model
├── inference.py                # Script for running inference with SpamGPT, including attention viz
├── ensemble_model.py           # Implementation of the Stacked Ensemble Model
├── ensemble_inference.py       # Script for running inference with the ensemble model
├── ensemble_evaluation.py      # Script for evaluating the ensemble model
├── eval_runner.py              # Simple runner script for ensemble_evaluation.py
├── checkpoints/                # Directory to save model checkpoints (e.g., best_model.pt for SpamGPT)
│   └── best_model.pt
├── training/                   # Directory related to encoder model training/output
│   └── model_output/           # Expected path for pre-trained encoder model files
│       └── (e.g., config.json, pytorch_model.bin)
├── ensemble_best_model.pt      # Saved checkpoint for the trained ensemble model
├── *.json                      # Data files (train_data_ham.json, test_data_spam.json, etc.)
│                               # and output files (ensemble_results.json, ensemble_test_evaluation.json)
├── gpu_monitor.py              # Utility for monitoring GPU usage
├── wandb/                      # Directory for Weights & Biases logs
├── nohup_*.out                 # Log files for background processes
├── ... (other utility scripts and notebooks)
```

## Dataset

### Format

The model expects data in JSON format with the following structure for each file (e.g., `train_data_ham.json`):

```json
{
    "dataset": [
        {
            "text": "Actual content of the email or message..."
        },
        {
            "text": "Another message..."
        }
        // ... more messages
    ]
}
```

- Separate files are used for ham and spam messages for training and testing (e.g., `train_data_ham.json`, `train_data_spam.json`).
- During processing for SpamGPT, texts are formatted with special tokens. For example: `<SOE>message content<EOE><SOP>` for inference, and `<SOE>message content<EOE><SOP><HAM><EOP>` (or `<SPAM>`) for training.

### Preparation

- Ensure your training and testing data (`train_data_ham.json`, `train_data_spam.json`, `test_data_ham.json`, `test_data_spam.json`) are in the root directory and follow the JSON format above.
- The `create_train_test.py` script might be used for splitting data, though its direct usage isn't detailed in the main workflows.

## Setup and Installation

### Dependencies

The project relies on several Python libraries. Key dependencies include:

- **PyTorch** (`torch`): For model building and training.
- **Transformers** (`transformers`): For using pre-trained encoder models.
- **TikToken** (`tiktoken`): For tokenization, especially for SpamGPT.
- **Scikit-learn** (`scikit-learn`): For evaluation metrics.
- **Pandas** (`pandas`): For data manipulation, used in evaluation.
- **Wandb** (`wandb`): For experiment tracking (optional, but integrated).
- **AIOHTTP** (`aiohttp`): For asynchronous HTTP requests (e.g., Discord webhooks).
- **python-dotenv**: For managing environment variables (e.g., webhook URLs).
- **tqdm**: For progress bars.
- **Matplotlib**: For plotting, including attention visualization.
- **Psutil**: For system utilities like GPU monitoring.

`pyproject.toml` lists:
- `datasets>=3.5.0`
- `scikit-learn>=1.6.1`

The `requirements.txt` file contains a comprehensive list of packages from the development Conda environment. For a cleaner setup, you might want to create a minimal `requirements.txt` or an environment configuration file (e.g., `environment.yml` for Conda) based on the actual imports and `pyproject.toml`.

### Environment

1.  **Python Version**: `pyproject.toml` specifies `requires-python = ">=3.13"`. Ensure you have a compatible Python version.
2.  **Conda (Recommended)**:
    ```bash
    # It's good practice to create a dedicated environment
    conda create -n spamgpt_env python=3.13
    conda activate spamgpt_env
    ```
3.  **Install Dependencies**:
    If a minimal `requirements.txt` is created:
    ```bash
    pip install -r requirements_minimal.txt
    ```
    Alternatively, install key packages manually:
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers tiktoken scikit-learn pandas wandb aiohttp python-dotenv tqdm matplotlib psutil datasets
    ```
    Ensure CUDA is set up correctly if using GPU.

## Usage

### 1. Configuration (`config.py`)

Adjust model parameters for SpamGPT (e.g., `block_size`, `n_layer`, `n_head`, `n_embd`) in `config.py` before training.

```python
# Example from config.py
@dataclass
class SpamGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50263 # GPT-2 base + 6 special tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

### 2. Training SpamGPT

The `train.py` script handles the training of the SpamGPT model.

```bash
python train.py
```

- **Distributed Training (DDP)**: The script automatically detects if it's a DDP run via environment variables (usually set by `torchrun`).
  ```bash
  # Example for DDP with torchrun (adjust nproc_per_node)
  # torchrun --standalone --nproc_per_node=gpu train.py
  ```
- **Wandb**: If `wandb` is installed and configured, logs will be sent to your Wandb project ("spamGPT" by default).
- **Checkpoints**: Best model checkpoints are saved in the `checkpoints/` directory (e.g., `checkpoints/best_model.pt`).
- **Discord Notifications**: Set `DISCORD_WEBHOOK_URL` in a `.env` file to receive notifications.

### 3. Training/Setting up the Encoder Model

The ensemble model relies on a pre-trained encoder. The scripts expect this model to be located at a path like `./training/model_output/` (or `./training/output/` as seen in `eval_runner.py`). This directory should contain Hugging Face model files (e.g., `pytorch_model.bin`, `config.json`, `tokenizer.json`).

- If you have a custom-trained encoder, ensure its artifacts are placed in the expected directory structure.
- If using a standard Hugging Face model, you might need to download it and save it to this path, or adjust the paths in the ensemble scripts. The `ensemble_evaluation.py` script's `load_pretrained_models` function uses `AutoConfig.from_pretrained(encoder_path)` and `AutoTokenizer.from_pretrained(encoder_path)`.

### 4. Training the Ensemble Model (Meta-Learner)

The provided scripts (`ensemble_model.py`, `ensemble_inference.py`, `ensemble_evaluation.py`) define the `StackedEnsembleModel`. If the `ensemble_method` is set to `meta_learner`, this meta-learner itself needs to be trained.
The file `ensemble_best_model.pt` suggests a pre-trained ensemble model (including the meta-learner if used) is available.

- The current scripts focus on loading a *pre-trained* ensemble model. If you need to train the meta-learner for the ensemble, a separate training script for the ensemble model would be required, which isn't explicitly provided but would involve:
    1.  Getting frozen predictions (logits or probabilities) from the already trained SpamGPT and encoder models on a training set.
    2.  Training the `meta_learner` component of `StackedEnsembleModel` using these predictions as input features and the true labels as targets.
    3.  Saving the state of the `StackedEnsembleModel` (which would primarily be the weights of the `meta_learner`).

### 5. Inference

#### SpamGPT Inference

Use `inference.py` for running predictions with the trained SpamGPT model. The `run_inference` function can generate text, and `get_spam_ham_probabilities` can provide classification probabilities and visualize attention maps (saved as `attention_matrix.png`). This script is more for direct interaction or debugging of SpamGPT.

Example (programmatic usage, adapt as needed):
```python
# from inference import run_inference, get_spam_ham_probabilities, enc, special_tokens
# from spamGPT import SpamGPT
# from config import SpamGPTConfig

# model = SpamGPT(SpamGPTConfig())
# model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device)['model_state_dict']) # or from checkpoint
# model.to(device)
# model.eval()

# text = "Your sample text here"
# input_text = f"<SOE>{text}<EOE><SOP>"
# input_ids = torch.tensor([enc.encode(input_text, allowed_special={"<SOE>", "<EOE>", "<SOP>"})], dtype=torch.long, device=device)

# spam_prob, ham_prob, highest_token = get_spam_ham_probabilities(input_ids, model)
# print(f"Spam: {spam_prob:.4f}, Ham: {ham_prob:.4f}, Next Predicted Token Type: {highest_token}")
```

#### Ensemble Model Inference

Use `ensemble_inference.py` for classifying text using the trained ensemble model.

```bash
# Classify a single text string
python ensemble_inference.py --text "Your sample email text here" \
    --ensemble_checkpoint ensemble_best_model.pt \
    --encoder_model_path ./training/model_output \
    --decoder_model_path ./checkpoints/best_model.pt \
    --output_file single_prediction.json

# Classify texts from a file (one text per line)
python ensemble_inference.py --input_file sample_inputs.txt \
    --ensemble_checkpoint ensemble_best_model.pt \
    --encoder_model_path ./training/model_output \
    --decoder_model_path ./checkpoints/best_model.pt \
    --output_file batch_predictions.json \
    --batch_size 16
```
- **Arguments**:
    - `--ensemble_checkpoint`: Path to the saved ensemble model (e.g., `ensemble_best_model.pt`).
    - `--encoder_model_path`: Path to the pre-trained encoder model directory.
    - `--decoder_model_path`: Path to the trained SpamGPT model checkpoint.
    - `--text`: Single text string for classification.
    - `--input_file`: Path to a file with texts to classify (one per line).
    - `--output_file`: File to save JSON results.
    - `--device`: `cuda` or `cpu`. Auto-detects if not specified.
    - `--batch_size`: For `--input_file` processing.
- Results are saved in JSON format, including ensemble and individual model predictions and probabilities.

### 6. Evaluation

#### Ensemble Model Evaluation

The `ensemble_evaluation.py` script evaluates the ensemble model (and its constituent base models) on test data. The `eval_runner.py` script provides a convenient way to run this with default parameters.

**Using `eval_runner.py` (recommended for standard evaluation):**
```bash
python eval_runner.py
```
This script checks for required files (`ensemble_best_model.pt`, `test_data_ham.json`, `test_data_spam.json`, encoder model config, SpamGPT checkpoint) and then executes `ensemble_evaluation.py` with predefined arguments.

**Running `ensemble_evaluation.py` directly (for custom parameters):**
```bash
python ensemble_evaluation.py \
    --ensemble_checkpoint ensemble_best_model.pt \
    --encoder_model_path ./training/model_output \ # or ./training/output based on runner
    --decoder_model_path ./checkpoints/best_model.pt \
    --test_ham test_data_ham.json \
    --test_spam test_data_spam.json \
    --output_file ensemble_test_evaluation.json \
    --batch_size 16
```
- This script loads ham and spam test data, performs predictions, and computes metrics: Accuracy, Precision, Recall, F1-score, and a confusion matrix.
- Results, including detailed metrics for ensemble, encoder, and decoder, are saved to the specified output JSON file (e.g., `ensemble_test_evaluation.json`).

### 7. GPU Monitoring

A utility `gpu_monitor.py` is available for monitoring GPU usage.
```bash
python gpu_monitor.py
```

## Results

- **Training Logs**: If Wandb is enabled, find detailed logs and metrics on your Wandb dashboard. `nohup.out` or `nohup_training.out` may also contain console logs.
- **Model Checkpoints**:
    - SpamGPT: `checkpoints/best_model.pt`
    - Ensemble Model: `ensemble_best_model.pt`
- **Inference Outputs**:
    - `ensemble_inference.py`: Saved to the `--output_file` (e.g., `ensemble_predictions.json`, `ensemble_results.json`).
    - `inference.py` (attention maps): `attention_matrix.png`.
- **Evaluation Reports**:
    - `ensemble_evaluation.py`: Saved to the `--output_file` (e.g., `ensemble_test_evaluation.json`). This JSON file contains comprehensive metrics. `results.txt` and `spam_results.txt` might contain older or specific test results.

## License

Please add your project's license information here. For example:

```
This project is licensed under the MIT License - see the LICENSE.md file for details.
```
(If you have a `LICENSE.md` file, refer to it. Otherwise, choose a license like MIT, Apache 2.0, etc.)

---

This README provides a comprehensive guide to understanding, setting up, and using the SpamGPT project. For further details, refer to the source code and comments within the respective Python files.


