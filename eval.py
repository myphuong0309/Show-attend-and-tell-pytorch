import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from src.dataset import FlickrDataset
from src.utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import argparse

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*80)
    print("MODEL EVALUATION ON TEST SET")
    print("="*80)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    encoder = checkpoint['encoder'].to(device)
    decoder = checkpoint['decoder'].to(device)
    encoder.eval()
    decoder.eval()
    
    print(f"Device: {device}")
    print(f"Beam Size: {args.beam_size}\n")
    
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    test_loader = torch.utils.data.DataLoader(
        dataset=FlickrDataset(args.data_folder, 'test', word_map, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    
    references = list()
    hypotheses = list()
    
    print("Evaluating test dataset...")
    with torch.no_grad():
        for i, (image, caps, caplens) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            
            k = args.beam_size
            encoder_out = encoder(image)  
            batch_size = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
            num_pixels = encoder_out.size(1)
            
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
            
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
            seqs = k_prev_words
            top_k_scores = torch.zeros(k, 1).to(device)
            h,c = decoder.init_hidden_state(encoder_out)
            
            complete_seqs = list()
            complete_seqs_scores = list()
            
            step = 1
            while True:
                embeddings = decoder.embedding(k_prev_words).squeeze(1)
                awe, _ = decoder.attention(encoder_out, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                awe = gate * awe
                
                h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))
                
                h_norm = decoder.layer_norm1(h)
                out = decoder.fc1(h_norm)
                out = decoder.layer_norm2(out)
                out = decoder.relu(out)
                scores = decoder.fc2(out)
                scores = F.log_softmax(scores, dim=1)
                scores = top_k_scores.expand_as(scores) + scores
                
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
                    
                prev_word_inds = top_k_words // vocab_size
                next_word_inds = top_k_words % vocab_size
                
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
                
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
                    
                k -= len(complete_inds)
                if k == 0: break
                
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                
                if step > 50: break
                step += 1
                
            if len(complete_seqs_scores) > 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                seq = seqs[0][:].tolist()
                
            img_caps = [c for c in caps[0].tolist() if c not in {word_map['<start>'], word_map['<pad>']}]
            
            pred_caption = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            
            references.append([img_caps])
            hypotheses.append(pred_caption)
            
    bleu4 = corpus_bleu(references, hypotheses)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"   BLEU-1: {bleu1:.4f} ({bleu1*100:6.2f}%)")
    print(f"   BLEU-2: {bleu2:.4f} ({bleu2*100:6.2f}%)")
    print(f"   BLEU-3: {bleu3:.4f} ({bleu3*100:6.2f}%)")
    print(f"   BLEU-4: {bleu4:.4f} ({bleu4*100:6.2f}%) ← Primary Metric")
    print(f"{'='*80}\n")
    print(f"\\n")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image Captioning Model')
    parser.add_argument('--data_folder', default='data/processed', help='Folder with processed data')
    parser.add_argument('--checkpoint', default='outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar', 
                       help='Path to checkpoint')
    parser.add_argument('--word_map', default='data/processed/word2idx.json', help='Path to word map JSON')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for beam search')
    
    args = parser.parse_args()
    evaluate(args)