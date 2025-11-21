import torch 
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        
        resnet = models.resnet50(pretrained=True)  # Load a pre-trained ResNet-50 model
        
        modules = list(resnet.children())[:-2]  # Remove the last two layers (avgpool and fc)
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        for param in self.resnet.parameters(): 
            param.requires_grad = False  # Freeze the parameters
            
    def forward(self, images):
        out = self.resnet(images)
        out = self.adaptive_pool(out)  
        
        out = out.permute(0, 2, 3, 1)
        out = out.view(out.size(0), -1, out.size(3))
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
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
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
        
        # LSTMCell: input là (embedding + context vector)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        # Các lớp Linear để khởi tạo trạng thái hidden/cell state từ ảnh
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Cổng sigmoid (f_beta) điều chỉnh mức độ quan trọng của context vector
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Lớp output dự đoán từ (Linear layer)
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
        
        # Làm phẳng ảnh: (batch, 14, 14, 2048) -> (batch, 196, 2048)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # 1. Sắp xếp batch theo độ dài caption giảm dần (để xử lý hiệu quả)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # 2. Tạo Embedding cho từ vựng
        embeddings = self.embedding(encoded_captions)  # (batch, max_len, embed_dim)

        # 3. Khởi tạo LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # 4. Tạo Tensor chứa kết quả
        # Ta sẽ dự đoán từ t=1 đến hết (bỏ qua <start>), nên độ dài là max_len - 1
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # 5. Vòng lặp qua từng bước thời gian t
        for t in range(max(decode_lengths)):
            # Tại bước t, chỉ xử lý những caption chưa kết thúc
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Tính Attention
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            # Cổng Gate (Soft Attention)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # Chạy 1 bước LSTM
            # Input = Nối (Embedding từ t) + (Context Vector t)
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            
            # Dự đoán từ tiếp theo
            preds = self.fc(self.dropout_layer(h))  # (batch_size_t, vocab_size)
            
            # Lưu kết quả
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind