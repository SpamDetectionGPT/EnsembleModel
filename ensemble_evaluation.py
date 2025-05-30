#!/usr/bin/env python3
"""
Dedicated evaluation script for the trained ensemble model.
Computes Precision, Recall, F1, and Accuracy on the test dataset.
FIXED VERSION - Handles PyTorch 2.6 weights_only parameter correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import argparse
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import pandas as pd

# Import required components
from spamGPT import SpamGPT
from config import SpamGPTConfig
from inference import enc, special_tokens

sys.path.append('training')
from training.models import TransformerForSequenceClassification
from transformers import AutoConfig, AutoTokenizer


class EnsembleDataset(Dataset):
    """Dataset for ensemble model evaluation."""
    
    def __init__(self, ham_file: str, spam_file: str, encoder_tokenizer, decoder_tokenizer, max_length: int = 512):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        
        # Load ham data (label 0)
        with open(ham_file, 'r') as f:
            ham_data = json.load(f)
            for item in ham_data['dataset']:
                self.data.append(item['text'])
                self.labels.append(0)
        
        # Load spam data (label 1)
        with open(spam_file, 'r') as f:
            spam_data = json.load(f)
            for item in spam_data['dataset']:
                self.data.append(item['text'])
                self.labels.append(1)
        
        print(f"Loaded {len(self.data)} test samples:")
        print(f"  Ham: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Spam: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        
        # Tokenize for encoder model
        encoder_inputs = self.encoder_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare text for decoder model with special tokens
        decoder_text = f"<SOE>{text}<EOE><SOP>"
        allowed_special = {"<SOE>", "<EOE>", "<SOP>", "<EOP>", "<SPAM>", "<HAM>"}
        decoder_ids = self.decoder_tokenizer.encode(decoder_text, allowed_special=allowed_special)
        
        # Pad or truncate decoder input
        if len(decoder_ids) > self.max_length:
            decoder_ids = decoder_ids[:self.max_length]
        else:
            decoder_ids = decoder_ids + [0] * (self.max_length - len(decoder_ids))
        
        return {
            'encoder_input_ids': encoder_inputs['input_ids'].squeeze(0),
            'decoder_input_ids': torch.tensor(decoder_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


class StackedEnsembleModel(nn.Module):
    """Stacked ensemble combining encoder and decoder transformers."""
    
    def __init__(self, encoder_model, decoder_model, ensemble_method='weighted_average', 
                 encoder_weight=0.5, use_meta_learner=False, hidden_size=128):
        super().__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.ensemble_method = ensemble_method
        self.encoder_weight = encoder_weight
        self.decoder_weight = 1 - encoder_weight
        self.use_meta_learner = use_meta_learner
        
        # Freeze base models
        for param in self.encoder_model.parameters():
            param.requires_grad = False
        for param in self.decoder_model.parameters():
            param.requires_grad = False
        
        if self.use_meta_learner:
            # Meta-learner: a small neural network that learns to combine predictions
            self.meta_learner = nn.Sequential(
                nn.Linear(4, hidden_size),  # 2 models × 2 classes = 4 inputs
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, 2)  # 2 output classes
            )
    
    def get_encoder_predictions(self, encoder_input_ids):
        """Get predictions from encoder model."""
        self.encoder_model.eval()
        with torch.no_grad():
            logits = self.encoder_model(encoder_input_ids)
            probs = F.softmax(logits, dim=-1)
        return logits, probs
    
    def get_decoder_predictions(self, decoder_input_ids):
        """Get predictions from decoder model."""
        self.decoder_model.eval()
        with torch.no_grad():
            # Get model output
            logits = self.decoder_model(decoder_input_ids)
            
            # Find <SOP> positions
            sop_token_id = special_tokens["<SOP>"]
            batch_size = decoder_input_ids.size(0)
            
            # Initialize with zeros
            classification_logits = torch.zeros(batch_size, 2, device=decoder_input_ids.device)
            
            for i in range(batch_size):
                # Find <SOP> token position
                sop_positions = (decoder_input_ids[i] == sop_token_id).nonzero(as_tuple=True)[0]
                
                if len(sop_positions) > 0 and sop_positions[0] < len(decoder_input_ids[i]) - 1:
                    # Get logits at position after <SOP>
                    next_token_logits = logits[i, sop_positions[0], :]
                    
                    # Extract logits for <HAM> and <SPAM> tokens
                    ham_logit = next_token_logits[special_tokens["<HAM>"]]
                    spam_logit = next_token_logits[special_tokens["<SPAM>"]]
                    
                    classification_logits[i, 0] = ham_logit
                    classification_logits[i, 1] = spam_logit
            
            probs = F.softmax(classification_logits, dim=-1)
        
        return classification_logits, probs
    
    def forward(self, encoder_input_ids, decoder_input_ids):
        """Forward pass through ensemble."""
        # Get predictions from both models
        encoder_logits, encoder_probs = self.get_encoder_predictions(encoder_input_ids)
        decoder_logits, decoder_probs = self.get_decoder_predictions(decoder_input_ids)
        
        if self.ensemble_method == 'weighted_average':
            # Weighted average of probabilities
            ensemble_probs = (self.encoder_weight * encoder_probs + 
                            self.decoder_weight * decoder_probs)
            return ensemble_probs, encoder_probs, decoder_probs
        
        elif self.ensemble_method == 'voting':
            # Hard voting
            encoder_preds = torch.argmax(encoder_probs, dim=-1)
            decoder_preds = torch.argmax(decoder_probs, dim=-1)
            
            # For binary classification, use majority voting
            ensemble_preds = (encoder_preds + decoder_preds) / 2.0
            ensemble_preds = (ensemble_preds > 0.5).long()
            
            # Convert back to probabilities
            ensemble_probs = F.one_hot(ensemble_preds, num_classes=2).float()
            return ensemble_probs, encoder_probs, decoder_probs
        
        elif self.ensemble_method == 'meta_learner':
            # Concatenate probabilities and pass through meta-learner
            combined_features = torch.cat([encoder_probs, decoder_probs], dim=-1)
            ensemble_logits = self.meta_learner(combined_features)
            ensemble_probs = F.softmax(ensemble_logits, dim=-1)
            return ensemble_probs, encoder_probs, decoder_probs
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


def load_pretrained_models(encoder_path, decoder_path, device):
    """Load pretrained encoder and decoder models."""
    # Load encoder model
    print("Loading encoder model...")
    encoder_config = AutoConfig.from_pretrained(encoder_path)
    encoder_model = TransformerForSequenceClassification(encoder_config)
    
    # Load encoder weights - use weights_only=True for model weights
    encoder_weights_path = os.path.join(encoder_path, "best_model.pt")
    if os.path.exists(encoder_weights_path):
        try:
            encoder_model.load_state_dict(torch.load(encoder_weights_path, map_location=device, weights_only=True))
            print(f"✓ Loaded encoder weights from {encoder_weights_path}")
        except:
            # Fallback to weights_only=False if needed
            encoder_model.load_state_dict(torch.load(encoder_weights_path, map_location=device, weights_only=False))
            print(f"✓ Loaded encoder weights from {encoder_weights_path} (fallback mode)")
    else:
        encoder_weights_path = os.path.join(encoder_path, "pytorch_model.bin")
        try:
            encoder_model.load_state_dict(torch.load(encoder_weights_path, map_location=device, weights_only=True))
            print(f"✓ Loaded encoder weights from {encoder_weights_path}")
        except:
            encoder_model.load_state_dict(torch.load(encoder_weights_path, map_location=device, weights_only=False))
            print(f"✓ Loaded encoder weights from {encoder_weights_path} (fallback mode)")
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    
    # Load decoder model
    print("Loading decoder model...")
    decoder_model = SpamGPT(SpamGPTConfig())
    
    # Load decoder weights - must use weights_only=False for checkpoints with metadata
    try:
        checkpoint = torch.load(decoder_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            decoder_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded decoder weights from checkpoint")
        else:
            decoder_model.load_state_dict(checkpoint)
            print(f"✓ Loaded decoder weights from state dict")
    except Exception as e:
        print(f"Error loading decoder model: {e}")
        raise
    
    return encoder_model, encoder_tokenizer, decoder_model, enc


def load_ensemble_model(ensemble_checkpoint_path, encoder_path, decoder_path, device):
    """Load the trained ensemble model."""
    # Load base models
    encoder_model, encoder_tokenizer, decoder_model, decoder_tokenizer = load_pretrained_models(
        encoder_path, decoder_path, device
    )
    
    # Load ensemble checkpoint - MUST use weights_only=False for full checkpoints
    print(f"Loading ensemble checkpoint from {ensemble_checkpoint_path}")
    try:
        checkpoint = torch.load(ensemble_checkpoint_path, map_location=device, weights_only=False)
        print("✓ Successfully loaded ensemble checkpoint")
    except Exception as e:
        print(f"❌ Error loading ensemble checkpoint: {e}")
        print("This might be due to PyTorch version compatibility issues.")
        raise
    
    # Create ensemble model
    ensemble_model = StackedEnsembleModel(
        encoder_model,
        decoder_model,
        ensemble_method=checkpoint.get('ensemble_method', 'meta_learner'),
        encoder_weight=checkpoint.get('encoder_weight', 0.5),
        use_meta_learner=True  # Assume meta-learner based on user request
    )
    
    # Load meta-learner weights if applicable
    if checkpoint.get('model_state_dict') is not None:
        try:
            ensemble_model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded meta-learner weights")
        except Exception as e:
            print(f"Warning: Could not load meta-learner weights: {e}")
            print("Continuing with randomly initialized meta-learner...")
    
    ensemble_model.to(device)
    ensemble_model.eval()
    
    print(f"✓ Ensemble model loaded successfully")
    print(f"  - Method: {ensemble_model.ensemble_method}")
    print(f"  - Encoder weight: {ensemble_model.encoder_weight:.4f}")
    print(f"  - Decoder weight: {ensemble_model.decoder_weight:.4f}")
    print(f"  - Uses meta-learner: {ensemble_model.use_meta_learner}")
    
    return ensemble_model, encoder_tokenizer, decoder_tokenizer


def evaluate_model(model, dataloader, device):
    """Comprehensive evaluation of the ensemble model."""
    model.eval()
    
    # Store predictions and ground truth
    all_ensemble_preds = []
    all_encoder_preds = []
    all_decoder_preds = []
    all_labels = []
    all_ensemble_probs = []
    all_encoder_probs = []
    all_decoder_probs = []
    misclassified_samples = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            encoder_ids = batch['encoder_input_ids'].to(device)
            decoder_ids = batch['decoder_input_ids'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']
            
            # Forward pass
            ensemble_probs, encoder_probs, decoder_probs = model(encoder_ids, decoder_ids)
            
            # Get predictions
            ensemble_preds = torch.argmax(ensemble_probs, dim=-1)
            encoder_preds = torch.argmax(encoder_probs, dim=-1)
            decoder_preds = torch.argmax(decoder_probs, dim=-1)
            
            # Store results
            all_ensemble_preds.extend(ensemble_preds.cpu().numpy())
            all_encoder_preds.extend(encoder_preds.cpu().numpy())
            all_decoder_preds.extend(decoder_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ensemble_probs.extend(ensemble_probs.cpu().numpy())
            all_encoder_probs.extend(encoder_probs.cpu().numpy())
            all_decoder_probs.extend(decoder_probs.cpu().numpy())
            
            # Collect misclassified samples for analysis
            for i in range(len(texts)):
                if ensemble_preds[i].item() != labels[i].item():
                    misclassified_samples.append({
                        'text': texts[i][:200] + '...' if len(texts[i]) > 200 else texts[i],
                        'true_label': 'spam' if labels[i].item() == 1 else 'ham',
                        'ensemble_pred': 'spam' if ensemble_preds[i].item() == 1 else 'ham',
                        'encoder_pred': 'spam' if encoder_preds[i].item() == 1 else 'ham',
                        'decoder_pred': 'spam' if decoder_preds[i].item() == 1 else 'ham',
                        'ensemble_confidence': ensemble_probs[i, ensemble_preds[i]].item(),
                        'encoder_confidence': encoder_probs[i, encoder_preds[i]].item(),
                        'decoder_confidence': decoder_probs[i, decoder_preds[i]].item(),
                    })
    
    # Convert to numpy arrays
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_encoder_preds = np.array(all_encoder_preds)
    all_decoder_preds = np.array(all_decoder_preds)
    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)
    all_encoder_probs = np.array(all_encoder_probs)
    all_decoder_probs = np.array(all_decoder_probs)
    
    return {
        'ensemble_preds': all_ensemble_preds,
        'encoder_preds': all_encoder_preds,
        'decoder_preds': all_decoder_preds,
        'labels': all_labels,
        'ensemble_probs': all_ensemble_probs,
        'encoder_probs': all_encoder_probs,
        'decoder_probs': all_decoder_probs,
        'misclassified': misclassified_samples
    }


def calculate_metrics(predictions, labels, model_name="Model"):
    """Calculate comprehensive metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_results(results):
    """Print comprehensive evaluation results."""
    ensemble_metrics = calculate_metrics(results['ensemble_preds'], results['labels'], "Ensemble")
    encoder_metrics = calculate_metrics(results['encoder_preds'], results['labels'], "Encoder")
    decoder_metrics = calculate_metrics(results['decoder_preds'], results['labels'], "Decoder")
    
    print("\n" + "="*70)
    print("                    EVALUATION RESULTS")
    print("="*70)
    
    # Create comparison table
    df = pd.DataFrame([ensemble_metrics, encoder_metrics, decoder_metrics])
    df = df.round(4)
    print("\nModel Performance Comparison:")
    print(df.to_string(index=False))
    
    # Detailed ensemble results
    print(f"\n{'='*50}")
    print("DETAILED ENSEMBLE MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {ensemble_metrics['accuracy']:.4f}")
    print(f"Precision: {ensemble_metrics['precision']:.4f}")
    print(f"Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"F1 Score:  {ensemble_metrics['f1']:.4f}")
    
    # Confusion Matrix
    cm_ensemble = confusion_matrix(results['labels'], results['ensemble_preds'])
    print(f"\nConfusion Matrix (Ensemble):")
    print(f"              Predicted")
    print(f"              Ham   Spam")
    print(f"Actual Ham    {cm_ensemble[0,0]:<5} {cm_ensemble[0,1]:<5}")
    print(f"Actual Spam   {cm_ensemble[1,0]:<5} {cm_ensemble[1,1]:<5}")
    
    # Model agreement analysis
    total_samples = len(results['labels'])
    encoder_decoder_agree = np.sum(results['encoder_preds'] == results['decoder_preds'])
    all_agree = np.sum(
        (results['ensemble_preds'] == results['encoder_preds']) & 
        (results['ensemble_preds'] == results['decoder_preds'])
    )
    
    print(f"\nModel Agreement Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Encoder-Decoder agree: {encoder_decoder_agree}/{total_samples} ({encoder_decoder_agree/total_samples*100:.1f}%)")
    print(f"All models agree: {all_agree}/{total_samples} ({all_agree/total_samples*100:.1f}%)")
    
    # Misclassification analysis
    print(f"\nMisclassification Analysis:")
    print(f"Total misclassified: {len(results['misclassified'])}")
    if len(results['misclassified']) > 0:
        print(f"\nTop 5 Misclassified Examples:")
        for i, sample in enumerate(results['misclassified'][:5]):
            print(f"\n{i+1}. Text: {sample['text']}")
            print(f"   True: {sample['true_label']} | Ensemble: {sample['ensemble_pred']} ({sample['ensemble_confidence']:.3f})")
            print(f"   Encoder: {sample['encoder_pred']} ({sample['encoder_confidence']:.3f}) | "
                  f"Decoder: {sample['decoder_pred']} ({sample['decoder_confidence']:.3f})")
    
    return ensemble_metrics, encoder_metrics, decoder_metrics, cm_ensemble


def save_results(results, ensemble_metrics, encoder_metrics, decoder_metrics, cm_ensemble, output_file):
    """Save detailed results to file."""
    detailed_results = {
        'summary': {
            'total_samples': len(results['labels']),
            'ham_samples': np.sum(results['labels'] == 0).item(),
            'spam_samples': np.sum(results['labels'] == 1).item(),
            'misclassified_count': len(results['misclassified'])
        },
        'ensemble_metrics': ensemble_metrics,
        'encoder_metrics': encoder_metrics,
        'decoder_metrics': decoder_metrics,
        'confusion_matrix': cm_ensemble.tolist(),
        'model_agreement': {
            'encoder_decoder_agree': int(np.sum(results['encoder_preds'] == results['decoder_preds'])),
            'all_models_agree': int(np.sum(
                (results['ensemble_preds'] == results['encoder_preds']) & 
                (results['ensemble_preds'] == results['decoder_preds'])
            ))
        },
        'misclassified_samples': results['misclassified'][:20]  # Save top 20 for analysis
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ensemble model")
    parser.add_argument("--ensemble_checkpoint", type=str, default="ensemble_best_model.pt",
                        help="Path to ensemble model checkpoint")
    parser.add_argument("--encoder_model_path", type=str, default="./training/output",
                        help="Path to encoder model directory")
    parser.add_argument("--decoder_model_path", type=str, default="./checkpoints/best_model.pt",
                        help="Path to decoder model checkpoint")
    parser.add_argument("--test_ham", type=str, default="test_data_ham.json",
                        help="Path to test ham data")
    parser.add_argument("--test_spam", type=str, default="test_data_spam.json",
                        help="Path to test spam data")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--output_file", type=str, default="ensemble_test_evaluation.json",
                        help="Output file for detailed results")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load ensemble model
        ensemble_model, encoder_tokenizer, decoder_tokenizer = load_ensemble_model(
            args.ensemble_checkpoint, args.encoder_model_path, 
            args.decoder_model_path, device
        )
        
        # Load test dataset
        print(f"\nLoading test dataset...")
        test_dataset = EnsembleDataset(
            args.test_ham, args.test_spam,
            encoder_tokenizer, decoder_tokenizer
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Run evaluation
        results = evaluate_model(ensemble_model, test_loader, device)
        
        # Print and save results
        ensemble_metrics, encoder_metrics, decoder_metrics, cm_ensemble = print_results(results)
        save_results(results, ensemble_metrics, encoder_metrics, decoder_metrics, cm_ensemble, args.output_file)
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())