import torch
import torch.nn as nn
from einops import rearrange

class MHA(nn.Module):
    def __init__(self, d_model, n_heads, drop_p, max_len_seq):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.drop_p = drop_p
        self.max_len_seq = max_len_seq

        self.fc_q = nn.Linear(d_model, d_model) 
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(drop_p) 
        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

        freqs = self.get_positional_encodings(max_len_seq, d_model // n_heads)
        self.register_buffer('cos_pos', freqs.cos())
        self.register_buffer('sin_pos', freqs.sin())

    def get_positional_encodings(self, seq_len, dim):
        theta = 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        freqs = pos / theta
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs

    def rotate_half(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def apply_rotary_emb(self, cos_pos, sin_pos, t, start_index=0):
        rot_dim = cos_pos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * cos_pos) + (self.rotate_half(t) * sin_pos)
        return torch.cat((t_left, t, t_right), dim=-1)

    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = rearrange(q, 'N W (H D) -> N H W D', H=self.n_heads) 
        k = rearrange(k, 'N W (H D) -> N H W D', H=self.n_heads)
        v = rearrange(v, 'N W (H D) -> N H W D', H=self.n_heads)

        seq_len = q.shape[-2]
        assert seq_len <= self.max_len_seq, f'Sequence length {seq_len} exceeds the maximum sequence length {self.max_len_seq}'

        cos_pos = self.cos_pos[:seq_len, :]
        sin_pos = self.sin_pos[:seq_len, :]

        q = self.apply_rotary_emb(cos_pos, sin_pos, q)
        k = self.apply_rotary_emb(cos_pos, sin_pos, k)

        #att_score = q @ k.transpose(-2, -1) / self.scale 
        #causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        #att_score.masked_fill_(causal_mask, float('-inf'))
        #att_weights = torch.softmax(att_score, dim=-1) 
        #att_weights = self.dropout(att_weights) 
        #att = att_weights @ v

        att = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, dropout_p=self.drop_p, is_causal=True, scale=self.scale)

        x = rearrange(att, 'N H W D -> N W (H D)') 
        x = self.fc_out(x)  

        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop_p):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc_gate = nn.Linear(dim, hidden_dim)
        self.fc_up = nn.Linear(dim, hidden_dim)
        self.fc_down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x_gate = self.dropout(nn.functional.silu(self.fc_gate(x)))
        x_up = self.dropout(self.fc_up(x))
        x_down = self.fc_down(x_gate * x_up)
        return x_down

class RMSLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSLayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # 그냥 sqrt한 것의 역수

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_p, max_len_seq):
        super().__init__()

        self.norm_mha = RMSLayerNorm(d_model)

        self.mha = MHA(d_model, n_heads, drop_p, max_len_seq)

        self.norm_ff = RMSLayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff, drop_p)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):

        residual = self.norm_mha(x)
        residual = self.mha(residual)
        residual = self.dropout(residual) 
        x = x + residual

        residual = self.norm_ff(x)
        residual = self.ff(residual)
        residual = self.dropout(residual)
        x = x + residual

        return x 

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len_seq, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.emb_in = nn.Embedding(vocab_size, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.Sequential(*[DecoderLayer(d_model, d_ff, n_heads, drop_p, max_len_seq) for _ in range(n_layers)])

        self.norm_out = RMSLayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x): 

        x = self.emb_in(x)
        x = self.dropout(x)

        x = self.layers(x) 

        x = self.norm_out(x) 
        x = self.fc_out(x) 

        return x

class LLaMA(nn.Module):
    def __init__(self, vocab_size, max_len_seq, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.decoder = Decoder(vocab_size, max_len_seq, n_layers, d_model, d_ff, n_heads, drop_p)

        self.n_heads = n_heads

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                m.weight.data *= 1/torch.sqrt(torch.tensor(n_layers*2)) 
            elif isinstance(m, nn.Embedding): 
                nn.init.normal_(m.weight, mean=0, std=0.02) 
        nn.init.normal_(self.decoder.fc_out.weight, mean=0, std=0.02) 

    def forward(self, x):
        out = self.decoder(x) 
        return out 