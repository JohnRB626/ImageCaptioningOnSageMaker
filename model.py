
import torch

from torch import nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights

class CaptioningModel(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nheads: int,
        nlayers: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.fc = nn.Linear(2048, d_model)
        self.encoder = encoder
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nheads, d_model, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        
        self.output = nn.Linear(d_model, vocab_size)
        
        
    def forward(self, imgs: Tensor, targets: Tensor) -> Tensor:
        device = targets.get_device()
        
        memory = self.encoder(imgs).unsqueeze(1)
        
        _, T = targets.shape
        pad_mask = targets == 0
        attn_mask = ~torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        
        targets = self.embedding(targets)
        targets = self.pos_enc(targets)
              
        features = self.decoder(targets, memory, tgt_mask=attn_mask, tgt_key_padding_mask=pad_mask)
        
        scores = self.output(features)
        
        return scores
        
class PositionalEncoding(nn.Module):
    
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        i = torch.arange(max_len).unsqueeze(1)
        div_term = torch.pow(10000, -torch.arange(0, embed_dim, 2) / embed_dim)
        
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(i * div_term)
        pe[0, :, 1::2] = torch.cos(i * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        output = x + self.pe[0, :x.size(1), :]
        output = self.dropout(output)
        return output