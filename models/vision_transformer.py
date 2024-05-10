import math
import torch
import torch.nn as nn


__all__ = ['VisionTransformer', 'vit_base', 'vit_large', 'vit_huge']


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, img_size=224, patch_size=16, embedding_dims=768, p_dropout=0.5):
        super(PatchEmbedding, self).__init__()
        self.embedding_dims = embedding_dims
        self.projection = nn.Conv2d(in_channels, self.embedding_dims, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dims), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, int(img_size / patch_size) ** 2 + 1, self.embedding_dims), requires_grad=True)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        B = x.size(0)

        out_embedding = self.projection(x) # (B, 768, 14, 14)
        out_embedding = out_embedding.view(B, self.embedding_dims, -1).permute(0, 2, 1) # (B, 196, 768)
        cls_tokens = self.cls_token.repeat(B, 1, 1) # (B, 1, 1)

        out = torch.cat([cls_tokens,out_embedding], dim=1) # (B, 197, 768)
        out += self.pos_embedding # (B, 197, 768)
        out = self.dropout(out) # (B, 197, 768)

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims=768, num_heads=8, p_dropout=0.5):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.head_dims = int(embedding_dims / num_heads)
        self.query = nn.Linear(embedding_dims, embedding_dims)
        self.key = nn.Linear(embedding_dims, embedding_dims)
        self.value = nn.Linear(embedding_dims, embedding_dims)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x):
        B = x.size(0)
        Q = self.query(x).view(B, self.num_heads, -1, self.head_dims) # (B, 8, 197, 96)
        K = self.key(x).view(B, self.num_heads, -1, self.head_dims) # (B, 8, 197, 96)
        V = self.value(x).view(B, self.num_heads, -1, self.head_dims) # (B, 8, 197, 96)

        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dims) # (B, 8, 197, 197)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (B, 8, 197, 197)
        attention_scores = self.dropout(attention_scores) # (B, 8, 197, 197)
        attention_outputs = torch.matmul(attention_scores, V) # (B, 8, 197, 96)

        out = attention_outputs.view(B, -1, self.embedding_dims) # (B, 197, 768)
        out = self.dropout(out) # (B, 197, 768)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims=768, num_heads=8, expansion_ratio=4, p_dropout=0.5):
        super(TransformerBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.LayerNorm(embedding_dims, eps=1e-6),
            MultiHeadSelfAttention(embedding_dims=embedding_dims, num_heads=num_heads, p_dropout=p_dropout),
            nn.Dropout(p=p_dropout)
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(embedding_dims, eps=1e-6),
            nn.Linear(embedding_dims, int(embedding_dims * expansion_ratio)),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(int(embedding_dims * expansion_ratio), embedding_dims),
            nn.Dropout(p=p_dropout)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out1 += x  # (B, 197, 768)

        out2 = self.block2(out1)
        out2 += out1  # (B, 197, 768)

        return out2


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, patch_size, embedding_dims, num_heads, p_dropout, num_blocks, expansion_ratio=4):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, img_size=img_size, patch_size=patch_size, embedding_dims=embedding_dims, p_dropout=p_dropout)

        transformer_blocks = []
        for _ in range(num_blocks):
            transformer_blocks.append(TransformerBlock(embedding_dims=embedding_dims, num_heads=num_heads, expansion_ratio=expansion_ratio, p_dropout=p_dropout))
        self.transformer_encoder = nn.Sequential(*transformer_blocks)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dims, eps=1e-6),
            nn.Linear(embedding_dims, num_classes)
        )

    def forward(self, x):
        out = self.patch_embedding(x) # (B, 197, 768)
        out = self.transformer_encoder(out) # (B, 197, 768)
        out = torch.mean(out, dim=1) # (B, 768)
        out = self.mlp_head(out) # (B, 1000)

        return out


def vit_base(**kwargs):
    return VisionTransformer(embedding_dims=768, num_heads=12, num_blocks=12, expansion_ratio=4, **kwargs)


def vit_large(**kwargs):
    return VisionTransformer(embedding_dims=1024, num_heads=16, num_blocks=24, expansion_ratio=4, **kwargs)


def vit_huge(**kwargs):
    return VisionTransformer(embedding_dims=1280, num_heads=16, num_blocks=32, expansion_ratio=4, **kwargs)


if __name__ == '__main__':
    img_size = 224
    in_channels = 3

    model = vit_base(in_channels=in_channels, num_classes=1000, img_size=224, patch_size=16, p_dropout=0.5)

    input = torch.randn(8, in_channels, img_size, img_size)

    output = model(input)
    print(output.shape)