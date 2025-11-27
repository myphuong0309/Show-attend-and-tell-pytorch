import os
import json 
from collections import Counter
from tqdm import tqdm
import string
import random
import argparse

random.seed(42)

def load_captions(captions_file):
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

def build_vocab(captions_mapping, min_word_freq=3):
    '''
    Build vocabulary from captions mapping
    Returns a word2idx dictionary
    '''
    word_counts = Counter()
    for image_id in captions_mapping:
        for caption in captions_mapping[image_id]:
            word_counts.update(caption.split())
            
    vocab = [w for w, c in word_counts.items() if c >= min_word_freq and w not in ['<start>', '<end>', '<pad>', '<unk>']]
    
    word2idx = {}
    word2idx['<pad>'] = 0
    word2idx['<start>'] = 1
    word2idx['<end>'] = 2
    word2idx['<unk>'] = 3
    
    for idx, word in enumerate(vocab):
        word2idx[word] = idx + 4
        
    print(f"Total vocabulary size: {len(word2idx)}")
    print(f"Special tokens: <pad>=0, <start>=1, <end>=2, <unk>=3")
    print(f"Regular words: {len(vocab)} (min frequency: {min_word_freq})")
    
    return word2idx

def split_dataset(captions_mapping, train_ratio=0.8, val_ratio=0.1):
    all_images = list(captions_mapping.keys())
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images ({len(train_images)/total*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({len(val_images)/total*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({len(test_images)/total*100:.1f}%)")
    
    return {'train': train_images, 'val': val_images, 'test': test_images}

def save_dataset(split_name, image_ids, captions_mapping, output_folder, images_folder):
    data = []
    missing_count = 0
    
    for img_id in image_ids:
        img_path = os.path.join(images_folder, img_id)
        if os.path.exists(img_path):
            data.append({
                'image_path': img_path,
                'captions': captions_mapping[img_id]
            })
        else:
            missing_count += 1
            
    output_file = os.path.join(output_folder, f'{split_name}_data.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"  {split_name}: Saved {len(data)} entries")
    if missing_count > 0:
        print(f"    Warning: {missing_count} images not found")
        
def main(args):
    print("="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    captions_file = os.path.join(args.data_folder, 'captions.txt')
    images_folder = os.path.join(args.data_folder, 'images')
    
    if not os.path.exists(captions_file):
        print(f"ERROR: Captions file '{captions_file}' not found!")
        return
    
    if not os.path.exists(images_folder):
        print(f"ERROR: Images folder '{images_folder}' not found!")
        return
    
    captions_mapping = load_captions(captions_file)
    print(f"\nLoaded captions for {len(captions_mapping)} images.")
    
    if len(captions_mapping) == 0:
        print("ERROR: No captions loaded. Check your captions.txt file format.")
        return
    
    total_captions = sum(len(caps) for caps in captions_mapping.values())
    print(f"Total captions: {total_captions}")
    print(f"Average captions per image: {total_captions/len(captions_mapping):.1f}")
    
    word2idx = build_vocab(captions_mapping, args.min_word_freq)
    with open(os.path.join(args.output_folder, 'word2idx.json'), 'w') as f:
        json.dump(word2idx, f, indent=2)
    print(f"\nVocabulary saved to {args.output_folder}/word2idx.json")
    
    splits = split_dataset(captions_mapping, args.train_ratio, args.val_ratio)
    
    print(f"\\nSaving datasets...")
    save_dataset('train', splits['train'], captions_mapping, args.output_folder, images_folder)
    save_dataset('val', splits['val'], captions_mapping, args.output_folder, images_folder)
    save_dataset('test', splits['test'], captions_mapping, args.output_folder, images_folder)
    
    print("\n" + "="*80)
    print("DATA PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Flickr8k Dataset')
    parser.add_argument('--data_folder', default='data', help='Folder with raw data')
    parser.add_argument('--output_folder', default='data/processed', help='Folder to save processed data')
    parser.add_argument('--min_word_freq', type=int, default=3, help='Minimum word frequency for vocabulary')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    main(args)