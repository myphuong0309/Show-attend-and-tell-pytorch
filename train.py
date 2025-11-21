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
encoder_lr = 1e-4
decoder_lr = 4e-4
epochs = 15
alpha_c = 1.0
embed_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()
    decoder.train()
    
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    
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
        
        clip_gradient(decoder_optimizer, 5.)
        if encoder_optimizer is not None:
            clip_gradient(encoder_optimizer, 5.)
        
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        
        acc = accuracy(scores, targets, 5)
        
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(acc, sum(decode_lengths))
        
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top5=top5accs))

def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()
    if encoder is not None:
        encoder.eval()
        
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    start = time.time()
    
    with torch.no_grad():
        for i, (images, captions, caplens) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)
            
            imgs = encoder(images)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, captions, caplens)
            targets = caps_sorted[:, 1:]
            
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            acc = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(acc, sum(decode_lengths))
            
            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), loss=losses, top5=top5accs))
    print(f"\n * Validation top-5 Acc {top5accs.avg:.3f}")
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
        encoder = Encoder()
        decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, dropout=dropout)
        
        # Encoder is frozen, so no optimizer needed
        encoder_optimizer = None
        decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
        
    else:
        print(f"Loading checkpoint from {checkpoint}...")
        checkpoint_data = torch.load(checkpoint)
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
    
    print("\nStarting training...")
    best_acc = 0.0
    epochs_since_improvement = 0
    patience = 5  # Stop if no improvement for 5 epochs
    
    for epoch in range(start_epoch, epochs):
        # Early stopping check
        if epochs_since_improvement >= patience:
            print(f"\nNo improvement for {patience} epochs. Stopping training.")
            break
            
        train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch
        )
        current_val_acc = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion
        )
        
        is_best = current_val_acc > best_acc
        if is_best:
            best_acc = current_val_acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        print(f"Epochs since improvement: {epochs_since_improvement}/{patience}")
        
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
        
if __name__ == '__main__':
    main()