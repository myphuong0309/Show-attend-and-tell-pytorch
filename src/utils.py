import torch
import os
import numpy as np

def init_embeddings(embeddings):
    '''
    Initialize embeddings with uniform distribution'''
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)
    
def load_embeddings(emb_file, word_map):
    '''
    Load pre-trained embeddings and create embedding tensor
    '''
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split()) - 1
    
    vocab = set(word_map.keys())
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embeddings(embeddings)
    
    print(f'Loading embeddings...')
    for line in open(emb_file, 'r'):
        line = line.split()
        emb_word = line[0]
        embedding = list(map(float, line[1:]))
        if emb_word in vocab:
            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
    
    return embeddings, emb_dim

def clip_gradient(optimizer, grad_clip):
    '''
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    '''
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)    
    
def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    '''
    Save model checkpoint
    '''
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer
    }
    
    directory = f'outputs/checkpoints'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = f'checkpoint_{data_name}.pth.tar'
    torch.save(state, os.path.join(directory, filename))
    
    if is_best:
        torch.save(state, os.path.join(directory, f'BEST_checkpoint_{data_name}.pth.tar'))
        
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, shrink_factor):
    '''
    Shrinks learning rate by a specified factor
    '''
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print(f"The new learning rate is {optimizer.param_groups[0]['lr']}\n")
    
def accuracy(scores, targets, k):
    '''
    Computes top-k accuracy
    '''
    batch_size = targets.size(0)
    
    _, ind = scores.topk(k, 1, True, True)
    
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    
    correct_total = correct.view(-1).float().sum()
    
    return correct_total.item() * (100.0 / batch_size)