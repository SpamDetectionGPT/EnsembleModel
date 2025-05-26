import torch
import os
import json
import numpy as np
import argparse
from transformers import AutoTokenizer
from training.models import TransformerForSequenceClassification, scaled_dot_product_attention

class AttentionExtractor:
    def __init__(self, model_path, device=None):
        """Initialize the attention extractor with a model path."""
        self.model_path = model_path
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                       "mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """Load the model and tokenizer."""
        # Load config
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dict to object for compatibility
        class Config:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        config = Config(**config_dict)
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print(f"Loaded tokenizer from {self.model_path}")
        except Exception as e:
            raise Exception(f"Failed to load tokenizer: {e}")
        
        # Instantiate model with the loaded config
        model = TransformerForSequenceClassification(config)
        
        # Load model weights from best_model.pt
        model_file = "best_model.pt"
        model_path = os.path.join(self.model_path, model_file)
        if not os.path.exists(model_path):
            model_file = "pytorch_model.bin"
            model_path = os.path.join(self.model_path, model_file)
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
        
        model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def get_attention_matrices(self, text, layer_idx=None):
        """
        Extract attention matrices from the model for the given text.
        
        Args:
            text (str): Input text to extract attention matrices for
            layer_idx (int, optional): Specific layer to extract from, or None for all layers
            
        Returns:
            dict: Dictionary with attention matrices and token information
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get attention matrices
        attention_matrices = self._extract_attention(input_ids, layer_idx)
        
        return {
            "tokens": tokens,
            "attention_matrices": attention_matrices,
            "text": text
        }
    
    def _extract_attention(self, input_ids, layer_idx=None):
        """
        Extract attention matrices from each layer.
        
        This implements a custom forward pass through the model to capture
        attention matrices at each layer.
        
        Args:
            input_ids (torch.Tensor): Input token ids
            layer_idx (int, optional): Specific layer to extract from, or None for all layers
            
        Returns:
            list: List of attention matrices for each layer
        """
        # First pass input through embeddings
        hidden_states = self.model.encoder.embeddings(input_ids)
        
        attention_matrices = []
        
        # Process each layer
        target_layers = (
            [self.model.encoder.layers[layer_idx]] if layer_idx is not None 
            else self.model.encoder.layers
        )
        
        for i, layer in enumerate(target_layers):
            # We need to compute attention weights for each layer
            
            # First apply layer norm
            norm_states = layer.layer_norm_1(hidden_states)
            
            # Then extract attention matrix by manually computing self-attention
            # We need to extract q, k, v for each attention head
            all_head_attentions = []
            
            for head in layer.attention.heads:
                q = head.q(norm_states)
                k = head.k(norm_states)
                v = head.v(norm_states)
                
                # Compute scaled dot product attention weights (before softmax)
                dim_k = k.size(-1)
                scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(dim_k, dtype=torch.float))
                
                # Apply softmax to get attention weights
                weights = torch.nn.functional.softmax(scores, dim=-1)
                
                # Store attention weights for this head
                all_head_attentions.append(weights.detach().cpu().numpy())
            
            # Store attention matrices for this layer
            attention_matrices.append(np.array(all_head_attentions))
            
            # Continue with the forward pass to get next layer's input
            hidden_states = layer(hidden_states)
        
        return attention_matrices
    
    def save_attention_data(self, data, output_file):
        """Save attention data to a file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            "tokens": data["tokens"],
            "text": data["text"],
            "attention_matrices": [
                [head.tolist() for head in layer] 
                for layer in data["attention_matrices"]
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f)
        
        print(f"Attention data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract attention matrices from a transformer model")
    parser.add_argument("--model_dir", type=str, default="./model_output", 
                        help="Directory containing the model files")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to extract attention for")
    parser.add_argument("--output_file", type=str, default="attention_data.json",
                        help="File to save attention data to")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Specific layer to extract attention from (default: all layers)")
    
    args = parser.parse_args()
    
    extractor = AttentionExtractor(args.model_dir, args.device)
    attention_data = extractor.get_attention_matrices(args.text, args.layer)
    extractor.save_attention_data(attention_data, args.output_file)


if __name__ == "__main__":
    main() 