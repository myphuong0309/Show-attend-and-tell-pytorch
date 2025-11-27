import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from src.model import Encoder, Decoder 
from src.dataset import FlickrDataset, CaptionCollate
from src.utils import *
import json
import os
import time
import argparse

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, alpha_c, grad_clip, device):
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
        
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, captions, caplens)
        
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        
        loss.backward()
        
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
        
        if i % 50 == 0:
            print(f'   Epoch {epoch:2d} [{i:4d}/{len(train_loader)}] | '
                  f'Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) | '
                  f'Top-5: {top5accs.val:5.1f}% (avg: {top5accs.avg:5.1f}%) | '
                  f'Time: {batch_time.val:.2f}s')
    
    print(f'\n   Epoch {epoch} Complete - Loss: {losses.avg:.4f} | Top-5 Acc: {top5accs.avg:.2f}%\n')
    return losses.avg, top5accs.avg

def validate(val_loader, encoder, decoder, criterion, alpha_c, device):
    decoder.eval()
    if encoder is not None:
        encoder.eval()
        
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    with torch.no_grad():
        for i, (images, captions, caplens) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            
            imgs = encoder(images)
            
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, captions, caplens)
            
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            acc = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(acc, sum(decode_lengths))
            
    print(f"   VALIDATION: Loss: {losses.avg:.4f} | Top-5 Acc: {top5accs.avg:.2f}%")
    return top5accs.avg

def main(args):
    global start_epoch, word2idx
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("IMAGE CAPTIONING WITH VISUAL ATTENTION - TRAINING")
    print("="*80)
    
    word2idx_file = os.path.join(args.data_folder, 'word2idx.json')
    with open(word2idx_file, 'r') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)
    
    print(f"\nCONFIGURATION:")
    print(f"   Device: {device}")
    print(f"   Vocabulary Size: {vocab_size}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Epochs: {args.epochs} (patience: {args.patience})")
    print(f"   Encoder Fine-tune: {args.fine_tune_encoder}")
    print(f"   Encoder LR: {args.encoder_lr}")
    print(f"   Decoder LR: {args.decoder_lr}")
    print(f"   Embed/Attention/Decoder Dim: {args.embed_dim}/{args.attention_dim}/{args.decoder_dim}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Alpha C: {args.alpha_c}")
    print(f"   Grad Clip: {args.grad_clip}")
    
    if args.checkpoint is None:
        start_epoch = 0
        encoder = Encoder(fine_tune=args.fine_tune_encoder)
        encoder_dim = encoder.encoder_dim
        decoder = Decoder(attention_dim=args.attention_dim, embed_dim=args.embed_dim, 
                         decoder_dim=args.decoder_dim, vocab_size=vocab_size, 
                         encoder_dim=encoder_dim, dropout=args.dropout)
        
        encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), 
                                      lr=args.encoder_lr, weight_decay=1e-5) if args.fine_tune_encoder else None
        decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), 
                                      lr=args.decoder_lr, weight_decay=1e-5)
    else:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False) 
        start_epoch = checkpoint_data['epoch'] + 1
        encoder = checkpoint_data['encoder']
        decoder = checkpoint_data['decoder']
        encoder_optimizer = checkpoint_data['encoder_optimizer']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'], label_smoothing=0.1).to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = FlickrDataset(
        args.data_folder, 'train', word2idx, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        normalize
    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True, 
        collate_fn=CaptionCollate(pad_idx=word2idx['<pad>']))
    
    val_dataset = FlickrDataset(
        args.data_folder, 'val', word2idx, 
        transform=transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            normalize]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True, 
        collate_fn=CaptionCollate(pad_idx=word2idx['<pad>']))
    
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80 + "\n")
    
    best_acc = 0.0
    epochs_since_improvement = 0
    
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='max', factor=0.5, patience=3)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(decoder_optimizer, T_0=10, T_mult=2)
    
    for epoch in range(start_epoch, args.epochs):
        if epochs_since_improvement >= args.patience:
            print("\n" + "="*80)
            print(f"⚠️  EARLY STOPPING after {epoch} epochs (no improvement for {args.patience} epochs)")
            print("="*80)
            break
        
        print(f"\n{'─'*80}")
        print(f"EPOCH {epoch}/{args.epochs-1} (Best Acc: {best_acc:.2f}%, No Improvement: {epochs_since_improvement}/{args.patience})")
        print(f"{'─'*80}")
        if epoch >= 5:
            ss_prob = min(0.5, (epoch - 5) * 0.025)
            decoder.set_ss_prob(ss_prob)
            print(f"   Scheduled Sampling Probability: {ss_prob:.3f}")
        train_loss, train_acc = train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, args.alpha_c, args.grad_clip, device)
        current_val_acc = validate(val_loader, encoder, decoder, criterion, args.alpha_c, device)
        
        current_lr = decoder_optimizer.param_groups[0]['lr']
        
        scheduler_plateau.step(current_val_acc)
        scheduler_cosine.step(epoch)
        new_lr = decoder_optimizer.param_groups[0]['lr']
        
        if new_lr < current_lr:
            print(f"   Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            
        is_best = current_val_acc > best_acc
        if is_best:
            improvement = current_val_acc - best_acc
            best_acc = current_val_acc
            epochs_since_improvement = 0
            print(f"\n   NEW BEST MODEL! Acc: {best_acc:.2f}% (+{improvement:.2f}%)")
        else:
            epochs_since_improvement += 1
            print(f"   No improvement (Best: {best_acc:.2f}%)")
            
        save_checkpoint('flickr8k', epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, 0, is_best)
        
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE - Best Validation Acc: {best_acc:.2f}%")
    print("="*80)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument('--data_folder', default='data/processed', help='Folder with processed data')
    parser.add_argument('--checkpoint', default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--fine_tune_encoder', action='store_true', help='Fine-tune encoder')
    parser.add_argument('--encoder_lr', type=float, default=1e-5, help='Encoder learning rate')
    parser.add_argument('--decoder_lr', type=float, default=5e-4, help='Decoder learning rate (increased)')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs (increased)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (increased)')
    parser.add_argument('--alpha_c', type=float, default=1.0, help='Doubly stochastic attention regularization')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping threshold')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--attention_dim', type=int, default=512, help='Attention dimension')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Decoder LSTM dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    args = parser.parse_args()
    main(args)