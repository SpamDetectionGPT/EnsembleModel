import torch
from transformers import AutoConfig, AutoTokenizer
from models import TransformerForSequenceClassification  # Import from models.py


def main():
    # Configuration
    model_ckpt = "bert-base-uncased"
    text = "As the aircraft becomes lighter, it flies higher in air of lower density to maintain the same airspeed."
    num_labels = 3  # Example number of labels for classification

    # Load configuration and tokenizer
    print(f"Loading configuration for {model_ckpt}...")
    config = AutoConfig.from_pretrained(model_ckpt)
    print(f"Loading tokenizer for {model_ckpt}...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Add number of labels to config for the classifier head
    config.num_labels = num_labels

    # Tokenize input text
    # Note: The custom Embeddings layer expects raw input_ids without special tokens ([CLS], [SEP])
    # It dynamically creates position_ids starting from 0 for the given sequence length.
    # If you intend to use [CLS] for classification as in standard BERT,
    # you might need to adjust the Embeddings layer or add special tokens here.
    print("Tokenizing text...")
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids
    print(f"Input IDs shape: {input_ids.shape}")

    # Instantiate the model
    print("Instantiating the custom Transformer model...")
    model = TransformerForSequenceClassification(config)
    model.eval()  # Set model to evaluation mode

    # Perform inference
    print("Running inference...")
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model(input_ids)

    print(f"Output logits shape: {outputs.shape}")
    print(f"Output logits: {outputs}")


if __name__ == "__main__":
    main()
