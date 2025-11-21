#!/bin/bash

# Dừng script ngay lập tức nếu có lệnh bị lỗi (tránh chạy tiếp khi bước trước fail)
set -e

echo "======================================================="
echo "   PROJECT: IMAGE CAPTIONING WITH VISUAL ATTENTION     "
echo "   Pipeline: Process -> Train -> Evaluate -> Inference "
echo "======================================================="

# ---------------------------------------------------------
# BƯỚC 1: XỬ LÝ DỮ LIỆU (Data Preprocessing)
# ---------------------------------------------------------
echo -e "\n[Step 1/4] Processing Data..."

# Kiểm tra xem dữ liệu đã được xử lý chưa
if [ -f "data/processed/word2idx.json" ] && [ -f "data/processed/test_data.json" ]; then
    echo "Found processed data files. Skipping preprocessing step."
else
    echo "Running process_input.py..."
    python process_input.py
fi

# ---------------------------------------------------------
# BƯỚC 2: HUẤN LUYỆN MÔ HÌNH (Training)
# ---------------------------------------------------------
echo -e "\n[Step 2/4] Starting Training Loop..."

# Kiểm tra xem đã có checkpoint tốt nhất chưa
CHECKPOINT="outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar"

if [ -f "$CHECKPOINT" ]; then
    echo "Found existing checkpoint: $CHECKPOINT"
    echo "Resuming training or starting new? (Press Ctrl+C to stop if you don't want to train more)"
    # python train.py # Bỏ comment dòng này nếu muốn train tiếp
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
# BƯỚC 4: CHẠY THỬ NGHIỆM (Inference Demo)
# ---------------------------------------------------------
echo -e "\n[Step 4/4] Running Single Image Inference..."

# Tải ảnh mẫu từ internet nếu chưa có (Ví dụ: ảnh một cậu bé chơi bóng chày)
TEST_IMG="test_sample.jpg"
if [ ! -f "$TEST_IMG" ]; then
    echo "Downloading sample image..."
    # Link ảnh mẫu (Flickr)
    wget -O $TEST_IMG "https://live.staticflickr.com/2876/9732237312_3963ebc64a_z.jpg" --no-check-certificate
fi

# Chạy inference
python inference.py --img "$TEST_IMG" --model "$CHECKPOINT" --beam_size 5

echo -e "\n======================================================="
echo "   PIPELINE FINISHED!"
echo "   - Check 'outputs/attention_maps/' for visualization."
echo "======================================================="