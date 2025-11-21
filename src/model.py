import torch 
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, fine_tune=False):
        super(Encoder, self).__init__()
        
        # Load pretrained VGG19
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Remove classifier (only keep features)
        # VGG19 features output is 512 channels
        self.features = vgg19.features  
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # Freeze all layers by default
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Fine-tune last few conv layers if specified
        if fine_tune:
            # Unfreeze last 4 conv layers (from layer 28 onwards in VGG19)
            for module in list(self.features.children())[28:]:
                for param in module.parameters():
                    param.requires_grad = True
        
        self.encoder_dim = 512  # VGG19 outputs 512 channels
            
    def forward(self, images):
        out = self.features(images)  # (batch, 512, H, W)
        out = self.adaptive_pool(out)  # (batch, 512, 14, 14)
        
        out = out.permute(0, 2, 3, 1)  # (batch, 14, 14, 512)
        out = out.view(out.size(0), -1, out.size(3))  # (batch, 196, 512)
        return out
    
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  
        
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden)  
        
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  
        
        alpha = self.softmax(att)  
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        
        return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super(Decoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1) 
        
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)  
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        embeddings = self.embedding(encoded_captions)  # (batch, max_len, embed_dim)

        h, c = self.init_hidden_state(encoder_out)
    
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            
            preds = self.fc(self.dropout_layer(h))  # (batch_size_t, vocab_size)
            
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind