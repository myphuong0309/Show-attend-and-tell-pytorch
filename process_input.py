import os
import json 
from collections import Counter
from tqdm import tqdm
import string
import random

DATA_FOLDER = 'data'
IMAGES_FOLDER = os.path.join(DATA_FOLDER, 'images')
CAPTIONS_FILE = os.path.join(DATA_FOLDER, 'captions.txt')
OUTPUT_FOLDER = 'data/processed'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
def load_captions(captions_file):
    '''
    Read file captions.txt (image, caption)
    Returns a dictionary mapping image IDs to lists of captions. {image_id: [list_captions]}
    '''
    with open(captions_file, 'r') as f:
        lines = f.readlines()
    
    mapping = dict()
    
    start_index = 0
    if "image, caption" in lines[0].lower():
        start_index = 1
        
    for i in tqdm(range(start_index, len(lines)), desc="Loading captions"):
        line = lines[i].strip()
        if len(line) < 2:
            continue
        
        parts = line.split(',', 1)
        if len(parts) < 2:
            continue
        
        image_id = parts[0]
        caption = parts[1]
        
        table = str.maketrans('', '', string.punctuation)
        caption = caption.translate(table)
        
        caption_words = caption.lower().split()
        caption_words = [word for word in caption_words if len(word) > 0 and word.isalpha()]
        
        final_caption = '<start> ' + ' '.join(caption_words) + ' <end>'
        
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(final_caption)
    
    return mapping

def build_vocab(captions_mapping, min_word_freq=5):
    '''
    Build vocabulary from captions mapping
    Returns a word2idx dictionary
    '''
    word_counts = Counter()
    for image_id in captions_mapping:
        for caption in captions_mapping[image_id]:
            word_counts.update(caption.split())
        
    vocab = [w for w, c in word_counts.items() if c >= min_word_freq]
    
    word2idx = {k: v + 1 for v, k in enumerate(vocab)}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = len(word2idx)
    
    print(f"Total vocabulary size: {len(word2idx)}")
    
    return word2idx

def split_dataset(captions_mapping, train_ratio=0.8, val_ratio=0.1):
    '''
    Split dataset into train, val, test sets
    Returns three dictionaries for each split
    '''
    all_images = list(captions_mapping.keys())
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    return {'train': train_images, 'val': val_images, 'test': test_images}

def save_dataset(split_name, image_ids, captions_mapping):
    data = []
    for img_id in image_ids:
        img_path = os.path.join(IMAGES_FOLDER, img_id)
        if os.path.exists(img_path):
            data.append({
                'image_path': img_path,
                'captions': captions_mapping[img_id]
            })
        
    output_file = os.path.join(OUTPUT_FOLDER, f'{split_name}_data.json')
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Split '{split_name}': Saved {len(data)} entries to {output_file}")
    
def main():
    if not os.path.exists(CAPTIONS_FILE):
        print(f"Captions file '{CAPTIONS_FILE}' not found!")
        return
    
    captions_mapping = load_captions(CAPTIONS_FILE)
    print(f"Loaded captions for {len(captions_mapping)} images.")
    
    word2idx = build_vocab(captions_mapping, min_word_freq=5)
    with open(os.path.join(OUTPUT_FOLDER, 'word2idx.json'), 'w') as f:
        json.dump(word2idx, f)
        
    splits = split_dataset(captions_mapping)
    
    save_dataset('train', splits['train'], captions_mapping)
    save_dataset('val', splits['val'], captions_mapping)
    save_dataset('test', splits['test'], captions_mapping)
    
    print("Data processing completed.")
    
if __name__ == '__main__':
    main()
    