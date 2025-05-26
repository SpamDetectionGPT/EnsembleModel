# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig, AutoTokenizer
from tqdm.auto import tqdm  # For progress bars
import os  # For saving model
import wandb
import time
from dotenv import load_dotenv

# Import from local modules
from models import TransformerForSequenceClassification
from utils.helpers import (
    parse_train_args,
    get_device,
)  # Import argument parser and device helper
from data.preparation import prepare_data, load_separate_train_test_data  # Import data preparation function

load_dotenv()

# --- Device Detection Function ---
def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# --- Main Training Function ---
def main():
    args = parse_train_args()  # Use the imported argument parser
    device = get_device()  # Use the imported device helper
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project="spam-classification",
        config={
            "model": args.model_ckpt,
            "num_labels": args.num_labels,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "device": str(device)
        }
    )
    
    # Check if multiple GPUs are available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs!")
    else:
        num_gpus = 0
        print("No GPUs found, using CPU.")

    # --- Load Config and Tokenizer ---
    print(f"Loading config for {args.model_ckpt}...")
    try:
        config = AutoConfig.from_pretrained(args.model_ckpt)
        config.num_labels = args.num_labels  # Set from arguments
    except Exception as e:
        print(f"Failed to load config '{args.model_ckpt}'. Error: {e}")
        exit()

    print(f"Loading tokenizer for {args.model_ckpt}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    except Exception as e:
        print(f"Failed to load tokenizer '{args.model_ckpt}'. Error: {e}")
        exit()

    # --- Load and Prepare Dataset ---
    try:
        # Define paths to train and test JSON files
        train_ham_path = "data/train_data_ham.json"
        train_spam_path = "data/train_data_spam.json"
        test_ham_path = "data/test_data_ham.json"
        test_spam_path = "data/test_data_spam.json"

        print(f"Loading training and test data from separate files...")
        dataset = load_separate_train_test_data(
            train_ham_path=train_ham_path,
            train_spam_path=train_spam_path,
            test_ham_path=test_ham_path,
            test_spam_path=test_spam_path
        )

        # Prepare data using the loaded dataset
        train_dataloader, eval_dataloader = prepare_data(args, tokenizer, dataset)
    except Exception as e:
        print(f"Data preparation failed. Error: {e}")
        print(f"Detailed error: {str(e)}")
        exit()

    # --- Instantiate Model ---
    print("Instantiating the custom Transformer model...")
    model = TransformerForSequenceClassification(config)
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model)
    # if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
    #     model = torch.compile(model)
        
    model.to(device)

    # --- Optimizer and Loss Function ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()  # Standard for classification

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory created at {checkpoint_dir}")

    # Track best model performance
    best_accuracy = 0.0
    
    # --- Training Loop ---
    num_training_steps = args.epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    # Calculate checkpoint frequency (save every X batches)
    checkpoint_freq = min(500, len(train_dataloader) // 2)  # Save twice per epoch or every 500 batches
    global_step = 0

    print("Starting training...")
    model.train()  # Ensure model starts in training mode
    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Epoch": epoch + 1, "Loss": loss.item()})
            
            # Log batch metrics to wandb
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/global_step": global_step
            })
            
            # Save periodic checkpoint (regardless of performance)
            if global_step % checkpoint_freq == 0:
                periodic_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                print(f"\nSaving periodic checkpoint at step {global_step} to {periodic_checkpoint_path}")
                # Save checkpoint with all information to resume training
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }
                torch.save(checkpoint, periodic_checkpoint_path)
                # Keep only the 3 most recent checkpoints to save disk space
                checkpoint_files = sorted(
                    [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_step_")],
                    key=lambda x: int(x.split("_")[2].split(".")[0])
                )
                if len(checkpoint_files) > 3:
                    for old_file in checkpoint_files[:-3]:
                        os.remove(os.path.join(checkpoint_dir, old_file))

        avg_train_loss = total_loss / len(train_dataloader)
        print(
            f"\nEpoch {epoch + 1}/{args.epochs} - Average Training Loss: {avg_train_loss:.4f}"
        )

        # --- Validation ---
        if eval_dataloader:
            model.eval()  # Set model to evaluation mode
            total_eval_loss = 0
            correct_predictions = 0
            total_predictions = 0

            print("Running validation...")
            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids)
                    loss = loss_fn(outputs, labels)
                    total_eval_loss += loss.item()

                    # Calculate accuracy
                    logits = outputs
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)

            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            accuracy = (
                correct_predictions / total_predictions
                if total_predictions > 0
                else 0.0
            )
            print(
                f"Validation Loss: {avg_eval_loss:.4f} - Validation Accuracy: {accuracy:.4f}"
            )
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
                "val/loss": avg_eval_loss,
                "val/accuracy": accuracy
            })
            
            # Save the best model based on validation accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = os.path.join(output_dir, "best_model.pt")
                print(f"New best accuracy: {best_accuracy:.4f}! Saving model to {best_model_path}")
                # Save the best model
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                
                # Save a complete checkpoint too
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                    "loss": avg_eval_loss,
                }
                torch.save(checkpoint, best_checkpoint_path)
                
                # Log best model metrics to wandb
                wandb.log({
                    "best/accuracy": best_accuracy,
                    "best/val_loss": avg_eval_loss
                })
                
            model.train()  # Set back to train mode for next epoch
        else:
            print("Skipping validation as no evaluation dataloader is available.")

    # --- Save Final Model ---
    try:
        print(f"Saving final model to {output_dir}")
        
        # Save model state dict (unwrap DataParallel if used)
        final_model_path = os.path.join(output_dir, "pytorch_model.bin")
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
        
        # Save final checkpoint with all training state
        final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
        checkpoint = {
            "epoch": args.epochs,
            "global_step": global_step,
            "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
        }
        torch.save(checkpoint, final_checkpoint_path)
            
        # Save config
        config.save_pretrained(output_dir)
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        print(f"Training finished. Best validation accuracy: {best_accuracy:.4f}")
        
        # Log final metrics to wandb
        wandb.log({
            "final/best_accuracy": best_accuracy,
            "final/epochs_completed": args.epochs,
            "final/total_steps": global_step
        })
        
    except Exception as e:
        print(f"Error saving model/config/tokenizer: {e}")

    print("Training finished.")
    wandb.finish()  # Properly close the wandb run


if __name__ == "__main__":
    main()
