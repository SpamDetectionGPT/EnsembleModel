#!/usr/bin/env python3
"""
Simple script to run ensemble model evaluation with default parameters.
Just run: python run_evaluation.py
"""

import subprocess
import sys
import os

def main():
    print("Running Ensemble Model Evaluation...")
    print("="*50)
    
    # Check if required files exist
    required_files = [
        "ensemble_best_model.pt",
        "test_data_ham.json", 
        "test_data_spam.json",
        "./training/output/config.json",
        "./checkpoints/best_model.pt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all model files and test data are available.")
        return
    
    print("✓ All required files found.")
    print("\nStarting evaluation...")
    
    # Run the evaluation script
    cmd = [
        sys.executable, 
        "ensemble_evaluation.py",
        "--ensemble_checkpoint", "ensemble_best_model.pt",
        "--encoder_model_path", "./training/output", 
        "--decoder_model_path", "./checkpoints/best_model.pt",
        "--test_ham", "test_data_ham.json",
        "--test_spam", "test_data_spam.json",
        "--batch_size", "16",
        "--output_file", "ensemble_test_evaluation.json"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✓ Evaluation completed successfully!")
        print("Results saved to: ensemble_test_evaluation.json")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        return
    except FileNotFoundError:
        print("\n❌ ensemble_evaluation.py not found. Please ensure the evaluation script is in the current directory.")
        return

if __name__ == "__main__":
    main()