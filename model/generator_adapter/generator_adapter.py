import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, Lq, D = query.shape
        Lk = key.shape[1]

        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.out_proj(out), attn

class PromptFusionAligner(nn.Module):
    def __init__(self, feat_dim, text_dim, num_heads=8, depth=6, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, feat_dim)
        self.layers = nn.ModuleList([
            CrossAttnBlock(feat_dim, num_heads, dropout)
            for _ in range(depth)
        ])

    def forward(self, style_feats_subset, text_embed):
        B, k, C, h, w = style_feats_subset.shape
        text_feats = self.text_proj(text_embed) # [B, L_text, C]

        style_tokens = style_feats_subset.view(B, k, C, h * w).permute(0, 3, 1, 2).reshape(B, k * h * w, C)

        for layer in self.layers:
            text_feats = layer(text_feats, style_tokens)

        text_token = text_feats.mean(dim=1, keepdim=True)
        text_fused_map = text_token.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, h, w)
        fused_style_feats = torch.cat([style_feats_subset, text_fused_map], dim=1)
        return fused_style_feats


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = ManualMultiHeadAttention(dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, style_tokens):
        attn_out, _ = self.cross_attn(self.norm1(x), style_tokens, style_tokens)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
