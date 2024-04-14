import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        print(f'ff input size: {x.size()}')
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        # dim_head: qkv output size of each head
        self.inner_dim = inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # to_q: (embed_dim, num_head, dim_head)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        print(f'attn input size: {x.size()}, to_qkv weight: {self.to_qkv}')

        print(self.inner_dim)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        print([i.size() for i in qkv])

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        print(q.size(), k.size(), v.size()) # (batch size 2, num_heads 12, num_patches + 1 65, d: dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        print(f'out: {out.size()}')
        
        res =  self.to_out(out)
        print(f'linear: {self.to_out}')

        print(f'result (out after linear): {res.size()}')

        return res

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        print(f'raw img: {img.size()}') # (B, c, h, w)

        x = self.to_patch_embedding(img) # (B, h*w/p^2, c*p^2) -> (B, h*w/p^2, d)
        
        print(f'raw patch dim: {self.patch_dim}')

        print(f'patch embeddings: {x.size()}')
        
        b, n, _ = x.shape # b: batch size, n: # patches

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        print(f'class tokens: {cls_tokens.size()}')

        x = torch.cat((cls_tokens, x), dim=1)
        
        print(f'class tokens + patch embeddings: {x.size()}')
        
        # print(self.pos_embedding[:, :(n + 1)].size(), self.pos_embedding.size())
        
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
    
if __name__ == '__main__':
    vit_b_32 = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024, # encoder layer/attention input/output size (Hidden Size D in the paper)
        depth = 12,
        heads = 12, # (Heads in the paper)
        dim_head = 64, # attention hidden size (seems be default, never change this)
        mlp_dim = 3072, # mlp layer hidden size (MLP size in the paper)
        dropout = 0.,
        emb_dropout = 0.
    )
    
    with torch.no_grad():
        r = torch.rand((2, 3, 256, 256))
        print(vit_b_32(r).size())
    
    import os
    torch.save(vit_b_32, './vit_l.pt')
    print(os.path.getsize('./vit_l.pt') / 1024**2)
    os.remove('./vit_l.pt')