import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from einops import rearrange, repeat

class TimeSformerLayer(nn.Module):
    """Temporal attention layer inspired by TimeSformer"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq_len, batch, hidden_dim]
        orig_shape = x.shape
        
        # Make sequence length divisible by t by padding
        t = 8
        seq_len = x.shape[0]
        if seq_len % t != 0:
            pad_len = t - (seq_len % t)
            # Pad with zeros to make sequence length divisible by t
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
            seq_len = x.shape[0]
        
        # Now seq_len is guaranteed to be divisible by t
        s = seq_len // t
        x = rearrange(x, '(t s) b d -> t (s b) d', t=t, s=s)  # Split into temporal chunks
        
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        
        # Rearrange back and trim to original sequence length if needed
        x = rearrange(x, 't (s b) d -> (t s) b d', b=orig_shape[1])
        if x.shape[0] > orig_shape[0]:
            x = x[:orig_shape[0]]
            
        return x

class MultiHeadFeatureFusion(nn.Module):
    """Attention-weighted feature fusion"""
    def __init__(self, video_dim, audio_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5

    def forward(self, video_feat, audio_feat):
        # video_feat: [seq_len, batch, video_dim]
        # audio_feat: [seq_len, batch, audio_dim]
        combined = torch.cat([video_feat, audio_feat], dim=-1)
        q = rearrange(self.query(video_feat), 's b (h d) -> h s b d', h=self.num_heads)
        k = rearrange(self.key(combined), 's b (h d) -> h s b d', h=self.num_heads)
        v = rearrange(self.value(combined), 's b (h d) -> h s b d', h=self.num_heads)
        
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'h s b d -> s b (h d)')
        return out

class MultimodalEncoder(nn.Module):
    def __init__(self, video_dim, audio_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        # Projections with expanded capacity
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim))
            
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim))
        
        # TimeSformer layers
        self.temporal_layers = nn.ModuleList([
            TimeSformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(2)])
        
        # Multi-head fusion
        self.feature_fusion = MultiHeadFeatureFusion(
            video_dim, audio_dim, hidden_dim)
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video_feat, audio_feat):
        # Enhanced feature projection
        video_feat = self.video_proj(video_feat)
        audio_feat = self.audio_proj(audio_feat)
        
        # Temporal modeling
        for layer in self.temporal_layers:
            video_feat = layer(video_feat)
            audio_feat = layer(audio_feat)
        
        # Attention-weighted fusion
        fused = self.feature_fusion(video_feat, audio_feat)
        
        # Cross-modal attention
        video_attn, _ = self.cross_attn(
            query=fused,
            key=torch.cat([video_feat, audio_feat], dim=0),
            value=torch.cat([video_feat, audio_feat], dim=0))
        
        return self.norm(fused + self.dropout(video_attn))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)]
        return x
    

class RLEnhancedDecoder(nn.Module):
    """Decoder with RL-friendly modifications"""
    def __init__(self, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu',
            batch_first=False)
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
        # RL-specific components
        self.value_head = nn.Linear(hidden_dim, 1)
        self.sample_prob = 0.2  # Initial exploration rate

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)
        
        # Value estimation for RL
        value = self.value_head(output.mean(dim=0))
        return output, value

class VideoAudioCaptioningModel(nn.Module):
    def __init__(self, vocab_size, video_dim, audio_dim, hidden_dim, 
                 num_encoder_layers=4, num_decoder_layers=4, num_heads=8, 
                 dropout=0.1, max_len=500):
        super().__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len)
        
        # Enhanced encoder
        self.multimodal_encoder = MultimodalEncoder(
            video_dim=video_dim,
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout)
        
        # RL-enhanced decoder
        self.decoder = RLEnhancedDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Special tokens
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.max_len = max_len

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                if 'value_head' in str(p):  # RL head needs smaller init
                    nn.init.normal_(p, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, video_feat, audio_feat, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # Encoder
        video_feat = video_feat.permute(1, 0, 2)
        audio_feat = audio_feat.permute(1, 0, 2)
        memory = self.multimodal_encoder(video_feat, audio_feat)
        
        # Prepare decoder inputs
        tgt = tgt.permute(1, 0)
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0))
        
        # Decoder with RL capability
        output, value = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask.to(tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask)
        
        logits = self.fc_out(output)
        return logits.permute(1, 0, 2), value

        
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence to prevent looking ahead"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate(self, video_feat, audio_feat, max_len=30, temperature=1.0, beam_size=3):
        """Generate captions using beam search"""
        self.eval()
        with torch.no_grad():
            # Encode inputs - directly permute the inputs since they already have batch dimension
            video_feat = video_feat.permute(1, 0, 2)  # [seq_len, batch, video_dim]
            audio_feat = audio_feat.permute(1, 0, 2)  # [seq_len, batch, audio_dim]
            memory = self.multimodal_encoder(video_feat, audio_feat)
            
            # Initialize beam search
            batch_size = video_feat.size(1)
            beams = [([self.sos_idx], 0)] * beam_size
            completed = []
            
            for _ in range(max_len):
                new_beams = []
                for seq, score in beams:
                    if seq[-1] == self.eos_idx:
                        completed.append((seq, score))
                        continue
                        
                    # Prepare input
                    tgt = torch.LongTensor(seq).unsqueeze(1).to(video_feat.device)
                    tgt_emb = self.embedding(tgt)  # [seq_len, 1, hidden]
                    tgt_emb = self.pos_encoder(tgt_emb)
                    
                    # Create mask
                    tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(video_feat.device)
                    
                    # Decode
                    output, _ = self.decoder(
                        tgt=tgt_emb,
                        memory=memory,
                        tgt_mask=tgt_mask)
                    
                    # Get next token probabilities
                    logits = self.fc_out(output[-1, :, :]) / temperature
                    probs = F.softmax(logits, dim=-1)
                    top_probs, top_tokens = probs.topk(beam_size, dim=1)
                    
                    # Expand beams
                    for i in range(beam_size):
                        token = top_tokens[0, i].item()
                        prob = top_probs[0, i].item()
                        new_seq = seq + [token]
                        new_score = score - math.log(prob + 1e-12)
                        new_beams.append((new_seq, new_score))
                
                # Keep top beams
                beams = sorted(new_beams, key=lambda x: x[1])[:beam_size]
            
            # Add any remaining beams to completed
            completed += beams
            
            # Select best sequence
            if not completed:
                return [self.sos_idx, self.eos_idx]
                
            best_seq = min(completed, key=lambda x: x[1])[0]
            return best_seq
