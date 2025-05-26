import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Dataset Loading and Tokenization ---

def load_json_data(file_path):
    """Helper function to load data from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)
    return data["dataset"]

def load_separate_train_test_data(train_ham_path, train_spam_path, test_ham_path, test_spam_path):
    """Load and combine separate train and test data files."""
    # Load training data
    train_ham_data = load_json_data(train_ham_path)
    train_spam_data = load_json_data(train_spam_path)
    
    # Load test data
    test_ham_data = load_json_data(test_ham_path)
    test_spam_data = load_json_data(test_spam_path)
    
    # Combine training data
    train_texts = []
    train_labels = []
    for item in train_ham_data:
        train_texts.append(item["text"])
        train_labels.append(0)
    for item in train_spam_data:
        train_texts.append(item["text"])
        train_labels.append(1)
    
    # Combine test data
    test_texts = []
    test_labels = []
    for item in test_ham_data:
        test_texts.append(item["text"])
        test_labels.append(0)
    for item in test_spam_data:
        test_texts.append(item["text"])
        test_labels.append(1)
    
    # Create DataFrames
    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    test_df = pd.DataFrame({"text": test_texts, "label": test_labels})
    
    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Create a DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
    
    return dataset_dict

# Keep the old function for backward compatibility
def combineandload_spamandham(ham_path, spam_path, test_size=0.2, random_state=42):
    """Legacy function for backward compatibility."""
    with open(ham_path, "r") as file:
        ham_data = json.load(file)
    with open(spam_path, "r") as file:
        spam_data = json.load(file)

    texts = []
    labels = []

    # Extract texts from the dataset array
    for item in ham_data["dataset"]:
        texts.append(item["text"])
        labels.append(0)
    for item in spam_data["dataset"]:
        texts.append(item["text"])
        labels.append(1)

    df = pd.DataFrame({"text": texts, "label": labels})

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create a DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset_dict


def prepare_data(args, tokenizer, dataset, cache_dir="cached_datasets"):
    """Tokenizes and prepares dataset splits for training and evaluation with caching support."""
    # Use the provided dataset directly
    raw_datasets = dataset

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a cache identifier based on the model and max length
    model_name = tokenizer.name_or_path.replace("/", "_")
    cache_file = f"{cache_dir}/tokenized_{model_name}_{args.max_length}.arrow"

    # --- Check for cached tokenized dataset ---
    if os.path.exists(cache_file):
        print(f"Loading tokenized dataset from cache: {cache_file}")
        try:
            tokenized_datasets = DatasetDict.load_from_disk(cache_file)
            print("Successfully loaded tokenized dataset from cache")
            # Skip to dataloader creation
        except Exception as e:
            print(f"Error loading cached dataset: {e}")
            print("Proceeding with tokenization...")
            tokenized_datasets = None
    else:
        print(f"No cached dataset found. Proceeding with tokenization...")
        tokenized_datasets = None

    # --- Input Validation ---
    # Check if train split exists
    if "train" not in raw_datasets:
        print("Error: Dataset must contain a 'train' split.")
        exit()

    # Ensure the necessary columns exist before mapping
    required_columns = {args.text_column, args.label_column}
    if not required_columns.issubset(raw_datasets["train"].column_names):
        print(
            f"Error: Dataset 'train' split missing required columns. Found: {raw_datasets['train'].column_names}, Required: {required_columns}"
        )
        exit()

    # --- Tokenization (if not loaded from cache) ---
    if tokenized_datasets is None:
        print("Tokenizing dataset...")

        # Define tokenization function using provided args
        def tokenize_function(examples):
            # add_special_tokens=False matches the model's Embeddings layer expectation
            return tokenizer(
                examples[args.text_column],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=False,
            )

        # Tokenize with progress tracking
        tokenized_datasets = raw_datasets.map(
            tokenize_function, batched=True, desc="Tokenizing"
        )

        # --- Formatting ---
        # Remove original text columns, set format to PyTorch tensors
        columns_to_remove = [
            col
            for col in raw_datasets["train"].column_names
            if col
            not in ["input_ids", "attention_mask", "token_type_ids", args.label_column]
        ]
        # Ensure label column is not accidentally removed if it shares name with another column
        if args.label_column in columns_to_remove:
            columns_to_remove.remove(args.label_column)

        tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
        # Rename the label column only if it's not already 'labels'
        if args.label_column != "labels":
            tokenized_datasets = tokenized_datasets.rename_column(
                args.label_column, "labels"
            )

        # Save tokenized dataset to cache
        print(f"Saving tokenized dataset to cache: {cache_file}")
        try:
            tokenized_datasets.save_to_disk(cache_file)
            print("Successfully saved tokenized dataset to cache")
        except Exception as e:
            print(f"Error saving tokenized dataset to cache: {e}")
            print("Continuing without saving cache...")

    # Set PyTorch format
    tokenized_datasets.set_format("torch")

    # --- Create DataLoaders ---
    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size
    )

    # Use validation or test split as needed - adjust split name if necessary
    eval_split = "test" if "test" in tokenized_datasets else "validation"
    if eval_split not in tokenized_datasets:
        print(
            f"Warning: No 'test' or 'validation' split found in dataset. Skipping evaluation."
        )
        eval_dataloader = None
    else:
        eval_dataset = tokenized_datasets[eval_split]
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    return train_dataloader, eval_dataloader


# --- Custom Dataset Class (if not using Hugging Face datasets) ---
class CustomTextDataset(Dataset):
    """A custom Dataset class for text classification.

    Args:
        texts (list[str]): A list of text samples.
        labels (list[int]): A list of corresponding labels.
        tokenizer: A Hugging Face tokenizer instance.
        max_length (int): Maximum sequence length for padding/truncation.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        # Squeeze to remove the batch dimension added by return_tensors='pt'
        # Prepare item dictionary expected by the training loop
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label)  # Ensure label is a tensor
        return item
