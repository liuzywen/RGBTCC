from nets.transformer_decoder_noPos import SingleAttention
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_


class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, drop_path=0.):
        super(Block, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.MHSA1 = SingleAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.mlp1 = Mlp(in_features=emb_dim, hidden_features=emb_dim*4, out_features=emb_dim)
        self.MHSA2 = SingleAttention(emb_dim=emb_dim, num_heads=num_heads)
        self.mlp2 = Mlp(in_features=emb_dim, hidden_features=emb_dim * 4, out_features=emb_dim)
        self.ProjR = nn.ModuleList([
                      nn.Linear(emb_dim, emb_dim),
                      nn.Linear(emb_dim * 7, emb_dim),
                      nn.Linear(emb_dim * 49, emb_dim)
                     ])
        self.ProjT = nn.ModuleList([
                      nn.Linear(emb_dim, emb_dim),
                      nn.Linear(emb_dim * 7, emb_dim),
                      nn.Linear(emb_dim * 49, emb_dim)
        ])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, rgb, thermal, count_token, i):
        size = [1, 7, 49]
        B, N, C = rgb.shape
        rgb = rgb.view(B, int(N / size[i]), size[i] * C).contiguous()
        thermal = thermal.view(B, int(N / size[i]), size[i] * C).contiguous()
        ProjR = self.ProjR[i]
        ProjT = self.ProjT[i]
        rgb = ProjR(rgb)
        thermal = ProjT(thermal)
        RGB_T = torch.cat((rgb, thermal), dim=1)
        token_fea = torch.cat((RGB_T, count_token), dim=1)
        token_fea = token_fea + self.drop_path(self.MHSA1(token_fea))
        token_fea = token_fea + self.mlp1(token_fea)
        token_fea = token_fea + self.drop_path(self.MHSA2(token_fea))
        token_fea = token_fea + self.mlp2(token_fea)
        token_fea = self.norm(token_fea)

        return token_fea


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, depth=3):
        super(MultiScaleAttention, self).__init__()
        self.MHSA_Block = nn.ModuleList([
            Block(emb_dim, num_heads) for i in range(depth)
        ])
        self.Proj = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim * 7),
            nn.Linear(emb_dim, emb_dim * 49)
        ])
        self.mlp = Mlp(in_features=emb_dim * 3, out_features=emb_dim)
        self.MLP = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            ),
            nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
        ])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, rgb, thermal, count_token):
        B, N, C = rgb.shape
        rgb_ca_t = torch.cat((rgb, thermal), dim=1)
        ca_tokens = torch.cat((rgb_ca_t, count_token), dim=1)
        size = [1, 7, 49]
        feas = []
        for i, block in enumerate(self.MHSA_Block):
            x = block(rgb, thermal, count_token, i)
            if i != 0:
                K = 2 * int(N / size[i])
                x = x[:, 0:K, :]
                count = x[:, -1, :].unsqueeze(1)
                x = self.Proj[i - 1](x)
                x = x.view(B, 2*N, C).contiguous()
                x = torch.cat((x, count), dim=1)
                x = torch.cat((x, ca_tokens), dim=2)
                x = self.MLP[i - 1](x)
            feas.append(x)

        token_fea = torch.cat((feas[0], feas[1]), dim=2)
        token_fea = torch.cat((token_fea, feas[2]), dim=2)

        token_fea = self.mlp(token_fea)

        return token_fea













