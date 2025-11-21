import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    """
    k = beam_size
    vocab_size = len(word_map)

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = transform(img).unsqueeze(0).to(device)

    encoder_out = encoder(image)  
    encoder_dim = encoder_out.size(-1)
    num_pixels = encoder_out.size(1)
    enc_image_size = int(num_pixels ** 0.5)  # sqrt of num_pixels (e.g., 196 -> 14)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  

    seqs = k_prev_words  
    top_k_scores = torch.zeros(k, 1).to(device) 
    
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device) 

    h, c = decoder.init_hidden_state(encoder_out)
    
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  

        awe, alpha = decoder.attention(encoder_out, h) 
        alpha = alpha.view(-1, enc_image_size, enc_image_size) 

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
        
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
            
        k -= len(complete_inds)
        if k == 0: break
        
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
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
        alphas = complete_seqs_alpha[i]
    else:
        seq = seqs[0][:].tolist()
        alphas = seqs_alpha[0].tolist()
        
    return seq, alphas

def visualize_attention(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with attention weights.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(4,5,1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Original", fontsize=12)

    for t in range(len(words)):
        if t > 50: break
        
        if words[t] == '<start>' or words[t] == '<end>' or words[t] == '<pad>':
            continue
        
        plt.subplot(4, 5, t + 2)
        
        plt.text(0, -1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        current_alpha = alphas[t]
        if smooth:
            alpha = skimage.transform.pyramid_expand(np.array(current_alpha), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(np.array(current_alpha), [14 * 24, 14 * 24])
        
        if smooth:
            plt.imshow(alpha, alpha=0.6)
        else:
            plt.imshow(alpha, alpha=0.8)
        
        plt.axis('off')
        
    output_path = "outputs/attention_maps/result.png"
    if not os.path.exists("outputs/attention_maps"):
        os.makedirs("outputs/attention_maps")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Attention visualization saved to {output_path}")
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend and Tell - Inference')
    
    parser.add_argument('--img', '--image', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model checkpoint')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON file', default='data/processed/word2idx.json')
    parser.add_argument('--beam_size', '-b', type=int, default=3, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    
    args = parser.parse_args()
    
    # Load word map (word2idx)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # Generate caption
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    
    # Convert indices to words
    words = [rev_word_map[ind] for ind in seq]
    caption = ' '.join(words)
    
    print(f"\nGenerated Caption: {caption}\n")
    
    # Visualize attention
    visualize_attention(args.img, seq, alphas, rev_word_map, smooth=args.smooth)
