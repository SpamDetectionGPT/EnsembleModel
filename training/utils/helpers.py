import torch
import argparse


def get_device():
    """Gets the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_train_args():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a Transformer for Sequence Classification"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="bert-base-uncased",
        help="Base model checkpoint for config/tokenizer",
    )
    # parser.add_argument(
    #     "--dataset_name",
    #     type=str,
    #     default="imdb",
    #     help="Name of the dataset (from Hugging Face or path to custom data)",
    # )
    # Add arguments for dataset splits, text/label columns if custom
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the dataset",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the dataset",
    )
    parser.add_argument(
        "--num_labels", type=int, default=2, help="Number of classes for classification"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length for tokenizer"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_output",
        help="Directory to save the trained model",
    )
    # Add other hyperparameters like weight decay, warmup steps etc. if needed
    return parser.parse_args()


# You could add other helpers here later, e.g., for logging, saving models
