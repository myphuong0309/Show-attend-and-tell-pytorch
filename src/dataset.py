import torch
from torch.utils.data import Dataset
import os
import json
import random
from PIL import Image
import torchvision.transforms as transforms

class FlickrDataset(Dataset):
    def __init__(self, data_folder, split, word2idx, transform=None):
        self.split = split
        self.transform = transform
        self.word2idx = word2idx
        
        file_path = os.path.join(data_folder, f'{split}_data.json')
        with open(file_path, 'r') as f:
            self.data = json.load(f)
    
    def __getitem__(self, index):
        item = self.data[index]
        image_path = item['image_path']
        captions = item['captions']
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.split == 'train':
            # Randomly select one caption during training for data augmentation
            caption = random.choice(captions)
        else:
            # Use first caption for validation/test
            caption = captions[0]
            
        tokens = caption.split()
        caption_indices = []
        
        for w in tokens:
            caption_indices.append(self.word2idx.get(w, self.word2idx['<unk>']))
            
        caption_tensor = torch.LongTensor(caption_indices)
        
        caplen = len(caption_indices)
        
        return image, caption_tensor, caplen
    
    def __len__(self):
        return len(self.data)


class CaptionCollate:
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        batch.sort(key=lambda x: x[2], reverse=True)
        
        images, captions, caplens = zip(*batch)
        
        images = torch.stack(images, 0)
        
        from torch.nn.utils.rnn import pad_sequence
        targets = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        lengths = torch.LongTensor(caplens)
        
        return images, targets, lengths
        