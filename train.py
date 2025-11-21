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
data_folder = 'data/processed'
checkpoint = None # 'outputs/checkpoints/checkpoint_flickr8k.pth.tar'
batch_size = 32
workers = 4 # Để 0 nếu chạy trên Windows
fine_tune_encoder = False 
encoder_lr = 1e-4
decoder_lr = 1e-4  # Reduced from 4e-4 for more stable training
epochs = 40
patience = 10  
alpha_c = 1.0 
grad_clip = 5.0  
embed_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (Phần print config giữ nguyên) ...

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
        
        # --- SỬA LẠI: Truyền thêm caplens ---
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(encoder_out, captions, caplens)
        
        # Targets là caption đã sort, bỏ từ <start>
        targets = caps_sorted[:, 1:]
        
        # Pack sequences (dùng decode_lengths trả về từ decoder cho an toàn)
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
            progress = (i / len(train_loader)) * 100
            print(f'   Epoch {epoch:2d} [{i:4d}/{len(train_loader)}] ({progress:5.1f}%) | '
                  f'Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) | '
                  f'Top-5: {top5accs.val:5.1f}% (avg: {top5accs.avg:5.1f}%) | '
                  f'Time: {batch_time.val:.2f}s')
    
    print(f'\n   Epoch {epoch} Complete - Loss: {losses.avg:.4f} | Top-5 Acc: {top5accs.avg:.2f}%\n')
    return losses.avg, top5accs.avg

def validate(val_loader, encoder, decoder, criterion):
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
            
            # --- SỬA LẠI: Truyền thêm caplens ---
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

def main():
    global start_epoch, word2idx
    
    print("="*80)
    print("IMAGE CAPTIONING WITH VISUAL ATTENTION - TRAINING")
    print("="*80)
    
    word2idx_file = os.path.join(data_folder, 'word2idx.json')
    with open(word2idx_file, 'r') as f:
        word2idx = json.load(f)
    vocab_size = len(word2idx)
    
    print(f"\n📊 CONFIGURATION:")
    print(f"   Device: {device}")
    print(f"   Vocabulary Size: {vocab_size}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs} (patience: {patience})")
    print(f"   Encoder: VGG19 (frozen={not fine_tune_encoder})")
    print(f"   Decoder LR: {decoder_lr}")
    print(f"   Embed/Attention/Decoder Dim: {embed_dim}/{attention_dim}/{decoder_dim}")
    print(f"   Dropout: {dropout}")
    print(f"   Alpha C: {alpha_c}")
    print(f"   Grad Clip: {grad_clip}")

    if checkpoint is None:
        start_epoch = 0
        encoder = Encoder(fine_tune=fine_tune_encoder)
        encoder_dim = encoder.encoder_dim 
        decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=encoder_dim, dropout=dropout)
        
        encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr, weight_decay=1e-5) if fine_tune_encoder else None
        decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr, weight_decay=1e-5)
        
    else:
        print(f"Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False) 
        start_epoch = checkpoint_data['epoch'] + 1
        encoder = checkpoint_data['encoder']
        decoder = checkpoint_data['decoder']
        encoder_optimizer = checkpoint_data['encoder_optimizer']
        decoder_optimizer = checkpoint_data['decoder_optimizer']
        
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Ignore padding tokens (index 0) when computing loss
    # Add label smoothing (0.1) to prevent overconfidence and improve generalization
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'], label_smoothing=0.1).to(device)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # ... (Phần DataLoader giữ nguyên) ...
    train_dataset = FlickrDataset(data_folder, 'train', word2idx, transform=transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=CaptionCollate(pad_idx=word2idx['<pad>']))
    
    val_dataset = FlickrDataset(data_folder, 'val', word2idx, transform=transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=CaptionCollate(pad_idx=word2idx['<pad>']))
    
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80 + "\n")
    
    best_acc = 0.0
    epochs_since_improvement = 0
    # Scheduler: reduce LR if no improvement after 5 epochs (increased from 3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='max', factor=0.5, patience=5)
    
    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement >= patience:
            print("\n" + "="*80)
            print(f"⚠️  EARLY STOPPING after {epoch} epochs (no improvement for {patience} epochs)")
            print("="*80)
            break
        
        print(f"\n{'─'*80}")
        print(f"EPOCH {epoch}/{epochs-1} (Best Acc: {best_acc:.2f}%, No Improvement: {epochs_since_improvement}/{patience})")
        print(f"{'─'*80}")
        
        train_loss, train_acc = train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch)
        current_val_acc = validate(val_loader, encoder, decoder, criterion)
        
        # Get current learning rate
        current_lr = decoder_optimizer.param_groups[0]['lr']
        
        scheduler.step(current_val_acc)
        new_lr = decoder_optimizer.param_groups[0]['lr']
        
        if new_lr < current_lr:
            print(f"   📉 Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
        
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
    main()