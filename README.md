# [PYTORCH] Show, Attend and Tell - Neural Image Caption Generation with Visual Attention

Here is my PyTorch implementation of the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (2015).

## Overview

This project implements an encoder-decoder architecture with attention mechanism for automatic image captioning:
- **Encoder**: Pre-trained ResNet-152 to extract rich 2048-dimensional image features
- **Decoder**: LSTM with soft attention and scheduled sampling to generate captions word-by-word
- **Attention**: Soft attention with gating mechanism to focus on relevant image regions

## Project Structure

```
.
├── data/
│   ├── images/              # Raw Flickr8k images (8091 images)
│   ├── captions.txt         # Image-caption pairs (5 captions per image)
│   └── processed/           # Processed data (auto-generated)
│       ├── train_data.json  # Training set (~6472 images)
│       ├── val_data.json    # Validation set (~809 images)
│       ├── test_data.json   # Test set (~810 images)
│       └── word2idx.json    # Vocabulary (3000+ words + special tokens)
├── outputs/
│   └── checkpoints/         # Model checkpoints
│       ├── checkpoint_flickr8k.pth.tar        # Latest
│       └── BEST_checkpoint_flickr8k.pth.tar   # Best validation
├── src/
│   ├── model.py            # Encoder (ResNet-152), Attention, Decoder
│   ├── dataset.py          # FlickrDataset, CaptionCollate
│   └── utils.py            # Training utilities, metrics
├── process_input.py        # Data preprocessing with argparse
├── train.py               # Training script with argparse
├── eval.py                # BLEU evaluation with argparse
├── run_pipeline.sh        # Automated pipeline script
└── requirements.txt       # Python dependencies
```

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/myphuong0309/Show-attend-and-tell-pytorch.git
cd Show-attend-and-tell-pytorch
```

2. Create a virtual environment (optional but recommended):
```bash
conda create -n env python=3.10
conda activate env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Flickr8k Dataset

1. Download the Flickr8k dataset:
   - Images: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Captions: Included in the dataset

2. Organize the data:
```
data/
├── images/           # Put all images here
└── captions.txt      # Format: image_name.jpg,caption
```

3. Format of `captions.txt`:
```
image,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way
1000268201_693b08cb0e.jpg,A girl going into a wooden building
...
```

## Usage

### Quick Start (Automated Pipeline)

Run the complete pipeline (preprocessing → training → evaluation → inference):

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Step-by-Step Usage

#### 1. Data Preprocessing

Process the Flickr8k dataset with custom parameters:

```bash
python process_input.py --data_folder data \
                        --output_folder data/processed \
                        --min_word_freq 3 \
                        --train_ratio 0.8 \
                        --val_ratio 0.1
```

**Arguments:**
- `--data_folder`: Folder containing `captions.txt` and `images/` (default: `data`)
- `--output_folder`: Where to save processed JSON files (default: `data/processed`)
- `--min_word_freq`: Minimum word frequency for vocabulary (default: 3)
- `--train_ratio`: Training split ratio (default: 0.8)
- `--val_ratio`: Validation split ratio (default: 0.1)

**Output:**
- `train_data.json`, `val_data.json`, `test_data.json` (80-10-10 split by default)
- `word2idx.json` (vocabulary mapping with special tokens)

#### 2. Training

Train the model with full control over hyperparameters:

```bash
python train.py --data_folder data/processed \
                --batch_size 32 \
                --decoder_lr 5e-4 \
                --encoder_lr 1e-5 \
                --epochs 60 \
                --patience 20 \
                --fine_tune_encoder \
                --embed_dim 512 \
                --attention_dim 512 \
                --decoder_dim 512 \
                --dropout 0.5 \
                --alpha_c 1.0 \
                --grad_clip 5.0
```

**Arguments:**
- `--data_folder`: Path to processed data (default: `data/processed`)
- `--checkpoint`: Resume from checkpoint (default: None)
- `--batch_size`: Batch size (default: 32)
- `--workers`: Data loading workers (default: 4)
- `--fine_tune_encoder`: Enable encoder fine-tuning (flag)
- `--encoder_lr`: Encoder learning rate (default: 1e-5)
- `--decoder_lr`: Decoder learning rate (default: 5e-4)
- `--epochs`: Training epochs (default: 60)
- `--patience`: Early stopping patience (default: 20)
- `--alpha_c`: Attention regularization weight (default: 1.0)
- `--grad_clip`: Gradient clipping threshold (default: 5.0)
- `--embed_dim`: Embedding dimension (default: 512)
- `--attention_dim`: Attention dimension (default: 512)
- `--decoder_dim`: Decoder LSTM dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.5)

**Features:**
- Scheduled sampling automatically activates from epoch 5
- Dual learning rate schedulers (ReduceLROnPlateau + CosineAnnealing)
- Label smoothing (0.1) for better generalization
- Enhanced data augmentation on-the-fly

**Output:**
- `checkpoint_flickr8k.pth.tar` (latest checkpoint)
- `BEST_checkpoint_flickr8k.pth.tar` (best validation accuracy)

#### 3. Evaluation

Compute BLEU scores on test set:

```bash
python eval.py --data_folder data/processed \
               --checkpoint outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar \
               --word_map data/processed/word2idx.json \
               --beam_size 3
```

**Arguments:**
- `--data_folder`: Folder with processed data (default: `data/processed`)
- `--checkpoint`: Path to checkpoint file (default: `outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar`)
- `--word_map`: Path to word map JSON (default: `data/processed/word2idx.json`)
- `--beam_size`: Beam search width (default: 3)

**Output:**
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores printed to console

#### 4. Inference on New Images

Generate captions for custom images (Note: `inference.py` not included in current workspace):

```bash
python inference.py --img path/to/image.jpg \
                   --model outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar \
                   --beam_size 5
```

**Arguments:**
- `--img`: Path to input image
- `--model`: Path to checkpoint file
- `--beam_size`: Beam search width (default: 3)
- `--dont_smooth`: Disable attention smoothing in visualization

**Output:**
- Generated caption printed to console
- Attention visualization saved to `outputs/attention_maps/`

## Implementation Details

### Scheduled Sampling
The model implements scheduled sampling to bridge the gap between training and inference:
- **Training**: Uses ground truth (teacher forcing) with probability (1 - ss_prob)
- **Inference**: Uses model's own predictions
- **Schedule**: Linear increase from 0% to 50% between epochs 5-25
- **Purpose**: Reduces exposure bias and improves BLEU scores significantly

### Dynamic Batching
The decoder handles variable-length sequences efficiently:
- Batch size decreases as sequences complete (reach `<end>` token)
- Optimizes computation by not processing completed sequences
- Properly handles tensor dimensions in scheduled sampling

### Two-Layer Output Architecture
Enhanced decoder output for better caption quality:
```
LSTM hidden (512) → LayerNorm → FC1 (512→256) → LayerNorm → ReLU → Dropout → FC2 (256→vocab)
```
This architecture provides:
- Better gradient flow via LayerNorm
- Non-linearity between output layers
- Reduced overfitting via dropout

## Results

The model shows progressive improvement through training optimizations:
- Baseline (ResNet-152, single-layer output): ~6-8% BLEU-4
- With scheduled sampling: Expected 15-25% BLEU-4 (training in progress)
- Target performance: 20-28% BLEU-4 on Flickr8k

Training metrics are logged every 50 batches showing loss and Top-5 accuracy.

## Model Architecture

### Encoder (ResNet-152)
- Pre-trained ResNet-152 with ImageNet weights
- Outputs 14×14×2048 feature maps via adaptive pooling
- Optional fine-tuning of later convolutional layers (layers 7-8)
- 2048-dimensional rich semantic features

### Attention Mechanism
- Soft attention over 196 spatial locations (14×14 grid)
- Learns to focus on relevant image regions for each word
- Gating mechanism (β) to control attention importance
- ReLU activation for attention computation

### Decoder (LSTM with Advanced Output)
- LSTM Cell with 512-dimensional hidden state
- Inputs: word embedding (512-dim) + attention-weighted features (2048-dim)
- Two-layer output projection with LayerNorm:
  - Layer 1: 512 → 256 with LayerNorm and ReLU
  - Layer 2: 256 → vocab_size
- Dropout on embeddings (0.5) and intermediate layers
- Scheduled sampling to reduce exposure bias
- Hidden/cell states initialized from mean image features

## References

- Original Paper: [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- Dataset: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Scheduled Sampling: [Scheduled Sampling for Sequence Prediction with RNNs](https://arxiv.org/abs/1506.03099)