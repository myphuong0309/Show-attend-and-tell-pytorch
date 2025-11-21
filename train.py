import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from src.model import Encoder, Decoder
from src.dataset import FlickrDataset, CaptionCollate
from src.utils import *
import json
import os
import time

## Configuration
data_folder = 'data/processed'  # folder with processed data
checkpoint = None
batch_size = 32
workers = 0
fine_tune_encoder = False  # Set to True to fine-tune last 2 ResNet blocks (improves BLEU but slower)
encoder_lr = 1e-4  # Used only if fine_tune_encoder=True
decoder_lr = 4e-4
epochs = 40 # Increased for better convergence
patience = 10  # Allow more exploration before stopping
alpha_c = 1.0 
grad_clip = 5.0  
embed_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"Device: {device}")
print(f"Batch Size: {batch_size}")
print(f"Fine-tune Encoder: {fine_tune_encoder}")
if fine_tune_encoder:
    print(f"Encoder Learning Rate: {encoder_lr}")
print(f"Decoder Learning Rate: {decoder_lr}")
print(f"Max Epochs: {epochs}")
print(f"Early Stopping Patience: {patience}")
print(f"Embed Dim: {embed_dim}, Attention Dim: {attention_dim}, Decoder Dim: {decoder_dim}")
print(f"Dropout: {dropout}, Alpha_c: {alpha_c}, Grad Clip: {grad_clip}")
print("="*80 + "\n")

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()
    decoder.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    start = time.time()
    
    for i, (images, captions, caplens) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        caplens = caplens.to(device)
        
        encoder_out = encoder(images)
        
        # Decoder returns (predictions, alphas) - only takes encoder_out and captions
        # Note: Model assumes fixed-length sequences (uses captions.size(1))
        scores, alphas = decoder(encoder_out, captions)
        
        # Targets are all words after <start>, up to <end>
        targets = captions[:, 1:]
        
        # Calculate decode lengths (actual lengths, not padded)
        decode_lengths = (caplens - 1).tolist()
        
        # Pack sequences to handle variable lengths
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss = criterion(scores, targets)
        
        # Doubly stochastic attention regularization
        att_regularization = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        loss += att_regularization
        
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        clip_gradient(decoder_optimizer, grad_clip)
        if encoder_optimizer is not None:
            clip_gradient(encoder_optimizer, grad_clip)
        
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        
        acc = accuracy(scores, targets, 5)
        
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(acc, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        
        if i % 50 == 0:  # Print every 50 batches
            print(f'[TRAIN] Epoch [{epoch}][{i}/{len(train_loader)}] | '
                  f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | '
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) | '
                  f'Top-5 Acc: {top5accs.val:.1f}% ({top5accs.avg:.1f}%)')
    
    print(f'\n[TRAIN SUMMARY] Epoch {epoch} | Avg Loss: {losses.avg:.4f} | Avg Top-5 Acc: {top5accs.avg:.1f}%\n')
    return losses.avg, top5accs.avg

def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()
    if encoder is not None:
        encoder.eval()
        
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    start = time.time()
    
    with torch.no_grad():
        for i, (images, captions, caplens) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            
            imgs = encoder(images)
            
            # Decoder returns (predictions, alphas) - only takes encoder_out and captions
            scores, alphas = decoder(imgs, captions)
            
            targets = captions[:, 1:]
            
            # Calculate decode lengths
            decode_lengths = (caplens - 1).tolist()
            
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            acc = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(acc, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()
            
            if i % 50 == 0:  # Print every 50 batches
                print(f'[VAL] [{i}/{len(val_loader)}] | '
                      f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) | '
                      f'Loss: {losses.val:.4f} ({losses.avg:.4f}) | '
                      f'Top-5 Acc: {top5accs.val:.1f}% ({top5accs.avg:.1f}%)')
    
    print(f"\n{'='*80}")
    print(f"[VALIDATION SUMMARY] Loss: {losses.avg:.4f} | Top-5 Acc: {top5accs.avg:.1f}%")
    print(f"{'='*80}\n")
    return top5accs.avg

def main():
    global start_epoch, word2idx
    
    word2idx_file = os.path.join(data_folder, 'word2idx.json')
    if not os.path.exists(word2idx_file):
        raise FileNotFoundError("word2idx.json not found in data folder. Please run data preprocessing (process_input.py) first.")
    
    with open(word2idx_file, 'r') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)
    
    if checkpoint is None:
        start_epoch = 0
        encoder = Encoder(fine_tune=fine_tune_encoder)
        encoder_dim = encoder.encoder_dim  # Get encoder_dim from encoder (512 for VGG19)
        decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=encoder_dim, dropout=dropout)
        
        # Create encoder optimizer if fine-tuning
        if fine_tune_encoder:
            encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr)
        else:
            encoder_optimizer = None
        decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        
    else:
        print(f"Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint, weights_only=False)
        start_epoch = checkpoint_data['epoch'] + 1
        encoder = checkpoint_data['encoder']
        decoder = checkpoint_data['decoder']
        encoder_optimizer = checkpoint_data['encoder_optimizer']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = FlickrDataset(data_folder, 'train', word2idx, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CaptionCollate(pad_idx=word2idx['<pad>'])
    )
    
    val_dataset = FlickrDataset(data_folder, 'val', word2idx, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CaptionCollate(pad_idx=word2idx['<pad>'])
    )
    
    print("\nStarting training...\n")
    best_acc = 0.0
    best_loss = float('inf')
    epochs_since_improvement = 0
    
    # Learning rate scheduler (optional but helps)
    if decoder_optimizer is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='max', factor=0.5, patience=3)
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Early stopping check
        if epochs_since_improvement >= patience:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING: No improvement for {patience} epochs.")
            print(f"Best Validation Accuracy: {best_acc:.1f}%")
            print(f"{'='*80}\n")
            break
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*80}")
        
        # Training
        train_loss, train_acc = train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch
        )
        
        # Validation
        current_val_acc = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion
        )
        
        # Learning rate scheduling
        if decoder_optimizer is not None:
            scheduler.step(current_val_acc)
        
        # Check if best
        is_best = current_val_acc > best_acc
        if is_best:
            best_acc = current_val_acc
            epochs_since_improvement = 0
            print(f"🎉 NEW BEST MODEL! Validation Acc: {best_acc:.1f}%")
        else:
            epochs_since_improvement += 1
            print(f"⏳ No improvement. Best: {best_acc:.1f}% | Patience: {epochs_since_improvement}/{patience}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch + 1} completed in {epoch_time/60:.1f} minutes")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"Val Acc: {current_val_acc:.1f}% | Best Val Acc: {best_acc:.1f}%")
        
        # Save checkpoint
        save_checkpoint(
            data_name='flickr8k',
            epoch=epoch,
            epochs_since_improvement=epochs_since_improvement,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            bleu4=0,
            is_best=is_best
        )
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"Best Validation Accuracy: {best_acc:.1f}%")
    print(f"Model saved to: outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar")
    print(f"{'='*80}\n")
        
if __name__ == '__main__':
    main()