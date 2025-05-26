import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import wandb

# Import decoder-only model components
from spamGPT import SpamGPT
from config import SpamGPTConfig
from inference import enc, special_tokens

# Import encoder-only model components
import sys
sys.path.append('training')
from training.models import TransformerForSequenceClassification
from transformers import AutoConfig, AutoTokenizer


wandb.init(
        project="Ensemble_Model",
        config={
            "model": "Ensemble",
        }
    )
    

class EnsembleDataset(Dataset):
    """Dataset for ensemble model that handles both encoder and decoder tokenization."""
    
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
        
        print(f"Loaded {len(self.data)} samples: {sum(1 for l in self.labels if l == 0)} ham, {sum(1 for l in self.labels if l == 1)} spam")
    
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
            'label': torch.tensor(label, dtype=torch.long)
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
        
        # Freeze base models initially
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
                    next_pos = sop_positions[0] + 1
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
            return ensemble_probs
        
        elif self.ensemble_method == 'voting':
            # Hard voting
            encoder_preds = torch.argmax(encoder_probs, dim=-1)
            decoder_preds = torch.argmax(decoder_probs, dim=-1)
            
            # For binary classification, use majority voting
            ensemble_preds = (encoder_preds + decoder_preds) / 2.0
            ensemble_preds = (ensemble_preds > 0.5).long()
            
            # Convert back to probabilities
            ensemble_probs = F.one_hot(ensemble_preds, num_classes=2).float()
            return ensemble_probs
        
        elif self.ensemble_method == 'meta_learner':
            # Concatenate probabilities and pass through meta-learner
            combined_features = torch.cat([encoder_probs, decoder_probs], dim=-1)
            ensemble_logits = self.meta_learner(combined_features)
            ensemble_probs = F.softmax(ensemble_logits, dim=-1)
            return ensemble_probs
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


class EnsembleTrainer:
    """Trainer for the ensemble model."""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-4, epochs=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        # Only train meta-learner if using it
        if model.use_meta_learner:
            self.optimizer = torch.optim.Adam(model.meta_learner.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
        else:
            # For weighted average, we can optionally learn the weights
            self.weight_param = nn.Parameter(torch.tensor([model.encoder_weight]))
            self.optimizer = torch.optim.Adam([self.weight_param], lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
    
    def train(self):
        """Train the ensemble model."""
        best_val_accuracy = 0.0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train() if self.model.use_meta_learner else self.model.eval()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                encoder_ids = batch['encoder_input_ids'].to(self.device)
                decoder_ids = batch['decoder_input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Update ensemble weights if not using meta-learner
                if not self.model.use_meta_learner and hasattr(self, 'weight_param'):
                    self.model.encoder_weight = torch.sigmoid(self.weight_param).item()
                    self.model.decoder_weight = 1 - self.model.encoder_weight
                
                # Forward pass
                ensemble_probs = self.model(encoder_ids, decoder_ids)
                
                # Calculate loss
                if self.model.use_meta_learner or self.model.ensemble_method == 'weighted_average':
                    loss = self.criterion(torch.log(ensemble_probs + 1e-8), labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(ensemble_probs, dim=-1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': train_loss / (train_total / labels.size(0)),
                    'acc': train_correct / train_total
                })
            
            train_accuracy = train_correct / train_total
            
            # Validation
            val_accuracy, val_metrics = self.evaluate(self.val_loader)
            
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            wandb.log({
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "val_precision": val_metrics['precision'],
                "val_recall": val_metrics['recall'],
                "val_f1": val_metrics['f1']
            })
            if not self.model.use_meta_learner:
                print(f"Encoder Weight: {self.model.encoder_weight:.4f}")
                print(f"Decoder Weight: {self.model.decoder_weight:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_checkpoint(f'ensemble_best_model.pt', epoch, val_metrics)
                print(f"✓ Saved new best model with validation accuracy: {val_accuracy:.4f}")
            
            print("-" * 50)
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                encoder_ids = batch['encoder_input_ids'].to(self.device)
                decoder_ids = batch['decoder_input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                ensemble_probs = self.model(encoder_ids, decoder_ids)
                predictions = torch.argmax(ensemble_probs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return accuracy, metrics
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if self.model.use_meta_learner else None,
            'encoder_weight': self.model.encoder_weight,
            'decoder_weight': self.model.decoder_weight,
            'ensemble_method': self.model.ensemble_method,
            'metrics': metrics
        }
        torch.save(checkpoint, path)


def load_pretrained_models(encoder_path, decoder_path, device):
    """Load pretrained encoder and decoder models."""
    # Load encoder model
    print("Loading encoder model...")
    encoder_config = AutoConfig.from_pretrained(encoder_path)
    encoder_model = TransformerForSequenceClassification(encoder_config)
    
    # Load encoder weights
    encoder_weights_path = os.path.join(encoder_path, "best_model.pt")
    if os.path.exists(encoder_weights_path):
        encoder_model.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    else:
        encoder_weights_path = os.path.join(encoder_path, "pytorch_model.bin")
        encoder_model.load_state_dict(torch.load(encoder_weights_path, map_location=device))
    
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    
    # Load decoder model
    print("Loading decoder model...")
    decoder_model = SpamGPT(SpamGPTConfig())
    
    # Load decoder weights
    checkpoint = torch.load(decoder_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        decoder_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        decoder_model.load_state_dict(checkpoint)
    
    return encoder_model, encoder_tokenizer, decoder_model, enc


def main():
    parser = argparse.ArgumentParser(description="Train stacked ensemble model")
    parser.add_argument("--encoder_model_path", type=str, default="./training/checkpoint/best_model.pt",
                        help="Path to encoder model directory")
    parser.add_argument("--decoder_model_path", type=str, default="./checkpoints/best_model.pt",
                        help="Path to decoder model checkpoint")
    parser.add_argument("--train_ham", type=str, default="train_data_ham.json",
                        help="Path to training ham data")
    parser.add_argument("--train_spam", type=str, default="train_data_spam.json",
                        help="Path to training spam data")
    parser.add_argument("--test_ham", type=str, default="test_data_ham.json",
                        help="Path to test ham data")
    parser.add_argument("--test_spam", type=str, default="test_data_spam.json",
                        help="Path to test spam data")
    parser.add_argument("--ensemble_method", type=str, default="weighted_average",
                        choices=["weighted_average", "voting", "meta_learner"],
                        help="Ensemble method to use")
    parser.add_argument("--encoder_weight", type=float, default=0.5,
                        help="Initial weight for encoder model (for weighted average)")
    parser.add_argument("--use_meta_learner", action="store_true",
                        help="Use a meta-learner to combine predictions")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained models
    encoder_model, encoder_tokenizer, decoder_model, decoder_tokenizer = load_pretrained_models(
        args.encoder_model_path, args.decoder_model_path, device
    )
    
    # Create ensemble model
    ensemble_model = StackedEnsembleModel(
        encoder_model, 
        decoder_model,
        ensemble_method=args.ensemble_method if not args.use_meta_learner else "meta_learner",
        encoder_weight=args.encoder_weight,
        use_meta_learner=args.use_meta_learner
    )
    
    # Create datasets
    print("\nLoading training data...")
    train_dataset = EnsembleDataset(
        args.train_ham, args.train_spam, 
        encoder_tokenizer, decoder_tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print("\nLoading test data...")
    test_dataset = EnsembleDataset(
        args.test_ham, args.test_spam,
        encoder_tokenizer, decoder_tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create trainer
    trainer = EnsembleTrainer(
        ensemble_model, train_loader, test_loader, 
        device, learning_rate=args.lr, epochs=args.epochs
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    test_accuracy, test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Ham   Spam")
    print(f"Actual Ham    {cm[0,0]:<5} {cm[0,1]:<5}")
    print(f"Actual Spam   {cm[1,0]:<5} {cm[1,1]:<5}")
    
    # Save final results
    results = {
        'test_accuracy': test_accuracy,
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'confusion_matrix': cm.tolist(),
        'encoder_weight': ensemble_model.encoder_weight,
        'decoder_weight': ensemble_model.decoder_weight,
        'ensemble_method': args.ensemble_method
    }
    
    with open('ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults saved to ensemble_results.json")


if __name__ == "__main__":
    main()