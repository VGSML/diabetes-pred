import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([d_model])))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute scaled dot-product attention
        attention_weights = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention weights to the values
        out = torch.matmul(attention_weights, V)

        return out, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of size (max_len, embed_dim) to store positional vectors
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)).unsqueeze(0)

        # Fill the matrix with positional values
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:, :div_term.size(1)-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch size dimension
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class PositionalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=5000):
        super(PositionalMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # position encoding
        x = self.positional_encoding(x)
        
        # apply multihead attention
        attn_output, _ = self.multihead_attention(x, x, x)

        # calculate weights and biases for Q, K, V
        q_weight = self.multihead_attention.in_proj_weight[:self.embed_dim, :]
        k_weight = self.multihead_attention.in_proj_weight[self.embed_dim:2*self.embed_dim, :]
        v_weight = self.multihead_attention.in_proj_weight[2*self.embed_dim:, :]
        
        q_bias = self.multihead_attention.in_proj_bias[:self.embed_dim]
        k_bias = self.multihead_attention.in_proj_bias[self.embed_dim:2*self.embed_dim]
        v_bias = self.multihead_attention.in_proj_bias[2*self.embed_dim:]

        # calculate Q, K, V
        Q = torch.matmul(x, q_weight.t()) + q_bias
        K = torch.matmul(x, k_weight.t()) + k_bias
        V = torch.matmul(x, v_weight.t()) + v_bias

        return attn_output, Q, K, V