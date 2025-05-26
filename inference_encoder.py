# inference.py
import torch
import argparse
from transformers import AutoConfig, AutoTokenizer
import os
import sys

# Add parent directory to path to allow importing from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from training.models import TransformerForSequenceClassification


def parse_inference_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Inference script for spam classification")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model_output",
        help="Directory containing the model files",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="best_model.pt",  # Use best model by default
        help="Model file to use for inference (best_model.pt, pytorch_model.bin, or checkpoint path)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to classify",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File containing texts to classify (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to write classification results to",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cuda, mps, cpu). Defaults to best available.",
    )
    return parser.parse_args()


def get_device(specified_device=None):
    """Get the best available device or the specified one."""
    if specified_device:
        if specified_device.lower() == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif specified_device.lower() == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif specified_device.lower() == "cpu":
            return torch.device("cpu")
        else:
            print(f"Warning: Specified device {specified_device} not available.")

    # Default to best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(model_dir, model_file):
    """Load the model, tokenizer, and config."""
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = AutoConfig.from_pretrained(model_dir)
    print(f"Loaded config from {model_dir}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Loaded tokenizer from {model_dir}")
    except Exception as e:
        raise Exception(f"Failed to load tokenizer: {e}")
    
    # Instantiate model with the loaded config
    model = TransformerForSequenceClassification(config)
    
    # Load model weights
    model_path = os.path.join(model_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Check if it's a checkpoint file
    if model_file.endswith(".pt"):
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:  # Full checkpoint
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from checkpoint {model_path}")
            if "accuracy" in checkpoint:
                print(f"Model checkpoint validation accuracy: {checkpoint['accuracy']:.4f}")
        else:  # Just the model state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model state from {model_path}")
    else:  # .bin file
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model from {model_path}")
    
    return model, tokenizer


def classify_text(text, model, tokenizer, device):
    """Classify a single text."""
    # Tokenize the input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Get both ham and spam probabilities
    ham_prob = probabilities[0][0].item()
    spam_prob = probabilities[0][1].item()
    
    # Map prediction to class
    result = {
        "text": text,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": confidence,
        "ham_probability": ham_prob,
        "spam_probability": spam_prob,
        "logits": logits[0].tolist()
    }
    
    return result


def batch_classify(texts, model, tokenizer, device, batch_size=8):
    """Classify a batch of texts."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the input batch
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).tolist()
            
        # Process each prediction in the batch
        for j, (text, prediction) in enumerate(zip(batch_texts, predictions)):
            confidence = probabilities[j][prediction].item()
            ham_prob = probabilities[j][0].item()
            spam_prob = probabilities[j][1].item()
            
            result = {
                "text": text,
                "prediction": "spam" if prediction == 1 else "ham",
                "confidence": confidence,
                "ham_probability": ham_prob,
                "spam_probability": spam_prob,
                "logits": logits[j].tolist()
            }
            results.append(result)
            
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")
            
    return results


def main():
    args = parse_inference_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(args.model_dir, args.model_file)
        model.to(device)
        
        # Process single text input
        if args.text:
            result = classify_text(args.text, model, tokenizer, device)
            print(f"Text: {result['text'][:100]}...")
            print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
            print(f"Ham probability: {result['ham_probability']:.4f}, Spam probability: {result['spam_probability']:.4f}")
            print(f"Logits: {result['logits']}")
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    f.write(f"Text: {result['text']}\n")
                    f.write(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})\n")
                    f.write(f"Ham probability: {result['ham_probability']:.4f}, Spam probability: {result['spam_probability']:.4f}\n")
                    f.write(f"Logits: {result['logits']}\n")
                print(f"Results written to {args.output_file}")
                
        # Process input file
        elif args.input_file:
            # Read input texts
            with open(args.input_file, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(texts)} texts from {args.input_file}")
            results = batch_classify(texts, model, tokenizer, device, args.batch_size)
            
            # Display and/or save results
            for result in results:
                print(f"Text: {result['text'][:100]}...")
                print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
                print(f"Ham probability: {result['ham_probability']:.4f}, Spam probability: {result['spam_probability']:.4f}")
                print("-" * 50)
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    for result in results:
                        f.write(f"Text: {result['text']}\n")
                        f.write(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})\n")
                        f.write(f"Ham probability: {result['ham_probability']:.4f}, Spam probability: {result['spam_probability']:.4f}\n")
                        f.write("-" * 50 + "\n")
                print(f"Results written to {args.output_file}")
        
        else:
            print("Error: No input provided. Use --text or --input_file to provide input.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 