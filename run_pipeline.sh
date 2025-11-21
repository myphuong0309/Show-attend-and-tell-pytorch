#!/bin/bash

set -e

echo "======================================================="
echo "   PROJECT: IMAGE CAPTIONING WITH VISUAL ATTENTION     "
echo "   Pipeline: Process -> Train -> Evaluate -> Inference "
echo "======================================================="

# ---------------------------------------------------------
# BƯỚC 1: XỬ LÝ DỮ LIỆU (Data Preprocessing)
# ---------------------------------------------------------
echo -e "\n[Step 1/4] Processing Data..."
echo "Running process_input.py..."
python process_input.py

# ---------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN MÔ HÌNH (Training)
# ---------------------------------------------------------
echo -e "\n[Step 2/4] Starting Training Loop..."

# Kiểm tra xem đã có checkpoint tốt nhất chưa
CHECKPOINT="outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar"

if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint: $CHECKPOINT"
    echo "Resuming training or starting new? (Press Ctrl+C to stop if you don't want to train more)"
    python train.py # Uncomment this line to continue training
    echo "Skipping training for this demo run (assuming model is ready)."
else
    echo "No checkpoint found. Starting training from scratch..."
    echo "Note: This process takes time. Ensure you have a GPU enabled."
    python train.py
fi

# ---------------------------------------------------------
# BƯỚC 3: ĐÁNH GIÁ (Evaluation - BLEU Score)
# ---------------------------------------------------------
echo -e "\n[Step 3/4] Evaluating Model on Test Set (BLEU-4)..."

if [ -f "$CHECKPOINT" ]; then
    python eval.py
else
    echo "ERROR: Cannot evaluate because no checkpoint was found!"
    echo "Please run training first."
    exit 1
fi

# ---------------------------------------------------------
# BƯỚC 4: CHẠY THỬ NGHIỆM (Inference Demo) - OPTIONAL
# ---------------------------------------------------------
# Uncomment below to run inference demo
# echo -e "\n[Step 4/4] Running Single Image Inference..."
# TEST_IMG="test_sample.jpg"
# if [ ! -f "$TEST_IMG" ]; then
#     if [ -d "data/images" ] && [ "$(ls -A data/images/*.jpg 2>/dev/null | head -1)" ]; then
#         FIRST_IMAGE=$(ls data/images/*.jpg | head -1)
#         cp "$FIRST_IMAGE" "$TEST_IMG"
#         echo "Using image from dataset: $FIRST_IMAGE"
#     fi
# fi
# if [ -f "$TEST_IMG" ]; then
#     python inference.py --img "$TEST_IMG" --model "$CHECKPOINT" --beam_size 5
# fi

echo -e "\n======================================================="
echo "   PIPELINE FINISHED!"
echo "   - Training and evaluation completed"
echo "   - To run inference: python inference.py --img <image> --model $CHECKPOINT"
echo "======================================================="