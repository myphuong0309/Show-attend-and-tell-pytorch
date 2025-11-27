#!/bin/bash

set -e

echo "======================================================="
echo "   PROJECT: IMAGE CAPTIONING WITH VISUAL ATTENTION     "
echo "   Pipeline: Process -> Train -> Evaluate              "
echo "======================================================="

# ---------------------------------------------------------
# Step 1: Data Preprocessing
# ---------------------------------------------------------
echo -e "\n[Step 1/4] Processing Data..."
echo "Running process_input.py..."
python process_input.py

# ---------------------------------------------------------
# Step 2: Training Loop
# ---------------------------------------------------------
echo -e "\n[Step 2/4] Starting Training Loop..."

CHECKPOINT="outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar"

if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint: $CHECKPOINT"
    echo "Resuming training or starting new? (Press Ctrl+C to stop if you don't want to train more)"
    # python train.py --fine_tune_encoder # Uncomment this if want to continue training
    echo "Skipping training for this demo run (assuming model is ready)."
else
    echo "No checkpoint found. Starting training from scratch..."
    echo "Note: This process takes time. Ensure you have a GPU enabled."
    python train.py --fine_tune_encoder
fi

# ---------------------------------------------------------
# Step 3: Evaluation - BLEU Score
# ---------------------------------------------------------
echo -e "\n[Step 3/4] Evaluating Model on Test Set (BLEU-4)..."

if [ -f "$CHECKPOINT" ]; then
    python eval.py
else
    echo "ERROR: Cannot evaluate because no checkpoint was found!"
    echo "Please run training first."
    exit 1
fi

echo -e "\n======================================================="
echo "   PIPELINE FINISHED!"
echo "======================================================="