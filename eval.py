import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from src.dataset import FlickrDataset
from src.utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import os

## Configuration
data_folder = 'data/processed'  # folder with processed data
checkpoint_file = 'outputs/checkpoints/BEST_checkpoint_flickr8k.pth.tar'
word_map_file = os.path.join(data_folder, 'word2idx.json')  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beam_size = 3

def evaluate(beam_size):
    print(f"Evaluating model from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    encoder = checkpoint['encoder'].to(device)
    decoder = checkpoint['decoder'].to(device)
    encoder.eval()
    decoder.eval()
    
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    test_loader = torch.utils.data.DataLoader(
        dataset=FlickrDataset(data_folder, 'test', word_map, transform=transforms.Compose([
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
            
            k = beam_size
            
            encoder_out = encoder(image)
            enc_image_size = encoder_out.size(1)
            encoder_dim = encoder_out.size(3)
            encoder_out = encoder_out.view(1, -1, encoder_dim)
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
                scores = decoder.fc(h)
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
    print(f"\nBLEU-4 score: {bleu4:.4f}\n")
    
if __name__ == '__main__':
    evaluate(beam_size)