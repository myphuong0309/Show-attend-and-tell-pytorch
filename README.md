# [PYTORCH] Show, Attend and Tell - Neural Image Caption Generation with Visual Attention

Here is my PyTorch implementation of the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (2015).

## Overview

This project implements an encoder-decoder architecture with attention mechanism for automatic image captioning:
- **Encoder**: Pre-trained ResNet-50 (frozen) to extract image features
- **Decoder**: LSTM with attention mechanism to generate captions word-by-word
- **Attention**: Soft attention to focus on relevant image regions while generating each word

## Features

- ✅ End-to-end training pipeline
- ✅ Beam search for better caption generation
- ✅ Attention visualization
- ✅ Early stopping with patience
- ✅ BLEU-4 score evaluation
- ✅ Automated pipeline script

## Project Structure

```
.
├── data/
│   ├── images/              # Place your images here
│   ├── captions.txt         # Image-caption pairs
│   └── processed/           # Processed data (auto-generated)
├── outputs/
│   ├── checkpoints/         # Model checkpoints
│   └── attention_maps/      # Attention visualizations
├── src/
│   ├── model.py            # Encoder, Decoder, Attention modules
│   ├── dataset.py          # PyTorch Dataset and DataLoader
│   └── utils.py            # Helper functions
├── process_input.py        # Data preprocessing
├── train.py               # Training script
├── eval.py                # Evaluation (BLEU score)
├── inference.py           # Generate captions for new images
├── run_pipeline.sh        # Automated pipeline
└── requirements.txt       # Dependencies
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)

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

2. Organize your data:
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

This will:
1. Process the dataset
2. Train the model
3. Evaluate on test set (BLEU-4)
4. Generate a sample caption with attention visualization

### Step-by-Step Usage

#### 1. Data Preprocessing

```bash
python process_input.py
```

This creates:
- `train_data.json`, `val_data.json`, `test_data.json` (80-10-10 split)
- `word2idx.json` (vocabulary mapping)

#### 2. Training

```bash
python train.py
```

**Training Configuration** (in `train.py`):
- `batch_size`: 32
- `encoder_lr`: 1e-4 (not used, encoder is frozen)
- `decoder_lr`: 4e-4
- `epochs`: 20
- `embed_dim`: 512
- `attention_dim`: 512
- `decoder_dim`: 512
- `dropout`: 0.5
- Early stopping patience: 5 epochs

The model saves:
- `checkpoint_flickr8k.pth.tar` (latest checkpoint)
- `BEST_checkpoint_flickr8k.pth.tar` (best validation accuracy)

#### 3. Evaluation

```bash
python eval.py
```

Computes BLEU-4 score on the test set using beam search.

#### 4. Inference on New Images

```bash
python inference.py --img path/to/image.jpg --model outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar --beam_size 5
```

**Arguments:**
- `--img`: Path to input image
- `--model`: Path to checkpoint file
- `--beam_size`: Beam search width (default: 3)
- `--dont_smooth`: Disable attention smoothing in visualization

**Output:**
- Generated caption printed to console
- Attention visualization saved to `outputs/attention_maps/`

## Model Architecture

### Encoder
- Pre-trained ResNet-50 (frozen weights)
- Removes final classification layers
- Outputs: 14×14×2048 feature map
- Adaptive pooling to 14×14 grid

### Attention Mechanism
- Soft attention over spatial image features
- Learns to focus on relevant regions for each word
- Gate mechanism (β) to control attention importance

### Decoder
- LSTM Cell with attention
- Input: word embedding + attention-weighted image features
- Hidden state initialized from mean image features
- Outputs vocabulary distribution at each timestep

## Training Details

- **Loss**: Cross-entropy + doubly stochastic attention regularization (α=1.0)
- **Optimizer**: Adam (decoder only)
- **Gradient Clipping**: 5.0
- **Encoder**: Frozen ResNet-50 features
- **Early Stopping**: Stops if validation accuracy doesn't improve for 5 epochs

## Results

Example captions will be displayed here after training.

## Citation

```bibtex
@article{xu2015show,
  title={Show, attend and tell: Neural image caption generation with visual attention},
  author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhudinov, Ruslan and Zemel, Rich and Bengio, Yoshua},
  journal={International conference on machine learning},
  pages={2048--2057},
  year={2015}
}
```

## License

MIT License

## Acknowledgments

- Original paper: [Show, Attend and Tell](https://arxiv.org/abs/1502.03044)
- Flickr8k dataset providers
- PyTorch team for the deep learning framework