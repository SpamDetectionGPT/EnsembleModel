import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_attention_data(file_path):
    """Load attention data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def plot_attention(attention_data, layer_idx=0, head_idx=0, output_file=None):
    """
    Plot attention matrix for a specific layer and head.
    
    Args:
        attention_data (dict): Attention data
        layer_idx (int): Layer index to visualize
        head_idx (int): Head index to visualize
        output_file (str, optional): Path to save the visualization
    """
    tokens = attention_data["tokens"]
    attention_matrix = np.array(attention_data["attention_matrices"][layer_idx][head_idx])
    
    # Determine the actual sequence length by removing padding tokens
    seq_len = len(tokens)
    for i, token in enumerate(tokens):
        if token == "[PAD]":
            seq_len = i
            break
    
    # Only plot the non-padding part of the matrix
    tokens = tokens[:seq_len]
    attention_matrix = attention_matrix[:seq_len, :seq_len]
    
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', ['#ffffff', '#0077cc'], N=256
    )
    
    # Plot the attention matrix
    plt.imshow(attention_matrix, cmap=cmap)
    plt.colorbar()
    
    # Add labels and make the plot readable
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=10)
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    
    plt.title(f'Attention Matrix - Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()

def prepare_for_backend(attention_data, output_file=None):
    """
    Prepare attention data for backend visualization.
    
    Args:
        attention_data (dict): Attention data
        output_file (str, optional): Path to save the prepared data
    
    Returns:
        dict: Processed data ready for backend visualization
    """
    tokens = attention_data["tokens"]
    
    # Determine the actual sequence length by removing padding tokens
    seq_len = len(tokens)
    for i, token in enumerate(tokens):
        if token == "[PAD]":
            seq_len = i
            break
    
    # Only keep the non-padding part of the data
    tokens = tokens[:seq_len]
    
    # Format data for visualization
    layers = []
    for layer_idx, layer_data in enumerate(attention_data["attention_matrices"]):
        heads = []
        for head_idx, head_data in enumerate(layer_data):
            # Extract and crop the attention matrix
            attention_matrix = np.array(head_data)[:seq_len, :seq_len].tolist()
            heads.append({
                "head_idx": head_idx,
                "attention": attention_matrix
            })
        
        layers.append({
            "layer_idx": layer_idx,
            "heads": heads
        })
    
    visualization_data = {
        "tokens": tokens,
        "text": attention_data["text"],
        "layers": layers
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(visualization_data, f)
        print(f"Visualization data saved to {output_file}")
    
    return visualization_data

def main():
    parser = argparse.ArgumentParser(description="Visualize attention matrices")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSON file containing attention data")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for visualization or processed data")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index to visualize (for plot mode)")
    parser.add_argument("--head", type=int, default=0,
                        help="Head index to visualize (for plot mode)")
    parser.add_argument("--mode", type=str, choices=["plot", "prepare"], default="plot",
                        help="Mode: plot (visualize) or prepare (for backend)")
    
    args = parser.parse_args()
    
    # Load attention data
    attention_data = load_attention_data(args.input_file)
    
    if args.mode == "plot":
        # Visualize attention matrix
        plot_attention(attention_data, args.layer, args.head, args.output_file)
    else:
        # Prepare data for backend
        prepare_for_backend(attention_data, args.output_file)

if __name__ == "__main__":
    main() 