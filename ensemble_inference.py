import torch
import argparse
import json
import os
import sys
from typing import List, Dict

# Import required components
from ensemble_model import StackedEnsembleModel, load_pretrained_models
from spamGPT import SpamGPT
from config import SpamGPTConfig
from inference import enc, special_tokens
from transformers import AutoTokenizer

sys.path.append('training')


def load_ensemble_model(ensemble_checkpoint_path, encoder_path, decoder_path, device):
    """Load the trained ensemble model."""
    # Load base models
    encoder_model, encoder_tokenizer, decoder_model, decoder_tokenizer = load_pretrained_models(
        encoder_path, decoder_path, device
    )
    
    # Load ensemble checkpoint
    checkpoint = torch.load(ensemble_checkpoint_path, map_location=device)
    
    # Create ensemble model
    ensemble_model = StackedEnsembleModel(
        encoder_model,
        decoder_model,
        ensemble_method=checkpoint.get('ensemble_method', 'weighted_average'),
        encoder_weight=checkpoint.get('encoder_weight', 0.5),
        use_meta_learner='meta_learner' in checkpoint.get('ensemble_method', '')
    )
    
    # Load meta-learner weights if applicable
    if checkpoint.get('model_state_dict') is not None:
        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
    
    ensemble_model.to(device)
    ensemble_model.eval()
    
    return ensemble_model, encoder_tokenizer, decoder_tokenizer


def classify_text(text: str, ensemble_model, encoder_tokenizer, decoder_tokenizer, device):
    """Classify a single text using the ensemble model."""
    # Tokenize for encoder
    encoder_inputs = encoder_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    encoder_input_ids = encoder_inputs['input_ids'].to(device)
    
    # Tokenize for decoder
    decoder_text = f"<SOE>{text}<EOE><SOP>"
    allowed_special = {"<SOE>", "<EOE>", "<SOP>", "<EOP>", "<SPAM>", "<HAM>"}
    decoder_ids = decoder_tokenizer.encode(decoder_text, allowed_special=allowed_special)
    
    # Pad or truncate
    max_length = 512
    if len(decoder_ids) > max_length:
        decoder_ids = decoder_ids[:max_length]
    else:
        decoder_ids = decoder_ids + [0] * (max_length - len(decoder_ids))
    
    decoder_input_ids = torch.tensor([decoder_ids], dtype=torch.long).to(device)
    
    # Get ensemble prediction
    with torch.no_grad():
        # Get individual model predictions
        encoder_logits, encoder_probs = ensemble_model.get_encoder_predictions(encoder_input_ids)
        decoder_logits, decoder_probs = ensemble_model.get_decoder_predictions(decoder_input_ids)
        
        # Get ensemble prediction
        ensemble_probs = ensemble_model(encoder_input_ids, decoder_input_ids)
        
        prediction = torch.argmax(ensemble_probs, dim=-1).item()
        confidence = ensemble_probs[0, prediction].item()
        
        # Get individual model predictions
        encoder_pred = torch.argmax(encoder_probs, dim=-1).item()
        encoder_conf = encoder_probs[0, encoder_pred].item()
        
        decoder_pred = torch.argmax(decoder_probs, dim=-1).item()
        decoder_conf = decoder_probs[0, decoder_pred].item()
    
    result = {
        'text': text,
        'ensemble_prediction': 'spam' if prediction == 1 else 'ham',
        'ensemble_confidence': confidence,
        'ensemble_ham_prob': ensemble_probs[0, 0].item(),
        'ensemble_spam_prob': ensemble_probs[0, 1].item(),
        'encoder_prediction': 'spam' if encoder_pred == 1 else 'ham',
        'encoder_confidence': encoder_conf,
        'encoder_ham_prob': encoder_probs[0, 0].item(),
        'encoder_spam_prob': encoder_probs[0, 1].item(),
        'decoder_prediction': 'spam' if decoder_pred == 1 else 'ham',
        'decoder_confidence': decoder_conf,
        'decoder_ham_prob': decoder_probs[0, 0].item(),
        'decoder_spam_prob': decoder_probs[0, 1].item(),
    }
    
    return result


def batch_classify(texts: List[str], ensemble_model, encoder_tokenizer, decoder_tokenizer, 
                  device, batch_size=16):
    """Classify multiple texts in batches."""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = []
        
        for text in batch_texts:
            result = classify_text(text, ensemble_model, encoder_tokenizer, 
                                 decoder_tokenizer, device)
            batch_results.append(result)
        
        results.extend(batch_results)
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with ensemble model")
    parser.add_argument("--ensemble_checkpoint", type=str, default="ensemble_best_model.pt",
                        help="Path to ensemble model checkpoint")
    parser.add_argument("--encoder_model_path", type=str, default="./training/model_output",
                        help="Path to encoder model directory")
    parser.add_argument("--decoder_model_path", type=str, default="./checkpoints/best_model.pt",
                        help="Path to decoder model checkpoint")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to classify")
    parser.add_argument("--input_file", type=str, default=None,
                        help="File containing texts to classify (one per line)")
    parser.add_argument("--output_file", type=str, default="ensemble_predictions.json",
                        help="Output file for predictions")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ensemble model
    print("Loading ensemble model...")
    ensemble_model, encoder_tokenizer, decoder_tokenizer = load_ensemble_model(
        args.ensemble_checkpoint, args.encoder_model_path, 
        args.decoder_model_path, device
    )
    
    # Process input
    if args.text:
        # Single text classification
        result = classify_text(args.text, ensemble_model, encoder_tokenizer, 
                             decoder_tokenizer, device)
        
        print("\nClassification Results:")
        print(f"Text: {result['text'][:100]}...")
        print(f"\nEnsemble Prediction: {result['ensemble_prediction']} "
              f"(confidence: {result['ensemble_confidence']:.4f})")
        print(f"  Ham probability: {result['ensemble_ham_prob']:.4f}")
        print(f"  Spam probability: {result['ensemble_spam_prob']:.4f}")
        
        print(f"\nEncoder Model:")
        print(f"  Prediction: {result['encoder_prediction']} "
              f"(confidence: {result['encoder_confidence']:.4f})")
        print(f"  Ham probability: {result['encoder_ham_prob']:.4f}")
        print(f"  Spam probability: {result['encoder_spam_prob']:.4f}")
        
        print(f"\nDecoder Model:")
        print(f"  Prediction: {result['decoder_prediction']} "
              f"(confidence: {result['decoder_confidence']:.4f})")
        print(f"  Ham probability: {result['decoder_ham_prob']:.4f}")
        print(f"  Spam probability: {result['decoder_spam_prob']:.4f}")
        
        # Save single result
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=4)
    
    elif args.input_file:
        # Batch classification
        print(f"\nReading texts from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} texts...")
        results = batch_classify(texts, ensemble_model, encoder_tokenizer, 
                               decoder_tokenizer, device, args.batch_size)
        
        # Print summary
        total = len(results)
        spam_count = sum(1 for r in results if r['ensemble_prediction'] == 'spam')
        ham_count = total - spam_count
        
        print(f"\nClassification Summary:")
        print(f"Total texts: {total}")
        print(f"Spam: {spam_count} ({spam_count/total*100:.1f}%)")
        print(f"Ham: {ham_count} ({ham_count/total*100:.1f}%)")
        
        # Model agreement analysis
        all_agree = sum(1 for r in results if 
                       r['encoder_prediction'] == r['decoder_prediction'] == r['ensemble_prediction'])
        encoder_decoder_agree = sum(1 for r in results if 
                                  r['encoder_prediction'] == r['decoder_prediction'])
        
        print(f"\nModel Agreement:")
        print(f"All models agree: {all_agree}/{total} ({all_agree/total*100:.1f}%)")
        print(f"Encoder-Decoder agree: {encoder_decoder_agree}/{total} "
              f"({encoder_decoder_agree/total*100:.1f}%)")
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total': total,
                    'spam_count': spam_count,
                    'ham_count': ham_count,
                    'all_models_agree': all_agree,
                    'encoder_decoder_agree': encoder_decoder_agree
                },
                'predictions': results
            }, f, indent=4)
        
        print(f"\nResults saved to {args.output_file}")
    
    else:
        print("Error: Please provide either --text or --input_file")
        return


if __name__ == "__main__":
    main()
