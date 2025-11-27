import torch 
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, fine_tune=False):
        super(Encoder, self).__init__()
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.encoder_dim = 2048  
        self.fine_tune(fine_tune)
        
    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out) 
        out = out.permute(0, 2, 3, 1) 
        return out 
    
    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            for c in list(self.resnet.children())[7:]:  
                for p in c.parameters():
                    p.requires_grad = True

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.encoder_att.weight)
        nn.init.xavier_uniform_(self.decoder_att.weight)
        nn.init.xavier_uniform_(self.full_att.weight)  
        
    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  
        att2 = self.decoder_att(decoder_hidden)  
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  
        alpha = self.softmax(att)  
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(Decoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.ss_prob = 0.0  
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm1 = nn.LayerNorm(decoder_dim)
        self.fc1 = nn.Linear(decoder_dim, decoder_dim // 2)
        self.layer_norm2 = nn.LayerNorm(decoder_dim // 2)
        self.fc2 = nn.Linear(decoder_dim // 2, vocab_size)
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        """Initialize embedding and output layer with careful initialization"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.xavier_uniform_(self.init_c.weight)
        nn.init.xavier_uniform_(self.f_beta.weight) 
        
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder_out))
        c = torch.tanh(self.init_c(mean_encoder_out))  
        return h, c
    
    def set_ss_prob(self, ss_prob):
        self.ss_prob = ss_prob
        
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.dropout_layer(self.embedding(encoded_captions)) 
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        prev_words = None
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            if self.training and t > 0 and self.ss_prob > 0 and prev_words is not None:
                use_sampling = torch.rand(batch_size_t).to(encoder_out.device) < self.ss_prob
                if use_sampling.any():
                    current_words = encoded_captions[:batch_size_t, t].clone()
                    current_words[use_sampling] = prev_words[:batch_size_t][use_sampling]
                    embedding_t = self.dropout_layer(self.embedding(current_words))
                else:
                    embedding_t = embeddings[:batch_size_t, t, :]
            else:
                embedding_t = embeddings[:batch_size_t, t, :]
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            lstm_input = torch.cat([embedding_t, attention_weighted_encoding], dim=1)
            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            h_norm = self.layer_norm1(h)
            out = self.fc1(self.dropout_layer(h_norm))
            out = self.layer_norm2(out)
            out = self.relu(out)
            preds = self.fc2(self.dropout_layer(out))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            if self.training:
                prev_words = preds.argmax(dim=1)
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind