import torch
import torch.nn as nn
from nets.pvt_v2 import pvt_v2_b3
from nets.transformer_decoder_noPos import transfmrerDecoder
from thop import profile, clever_format
from nets.MultiScaleAttention import MultiScaleAttention


class ThermalRGBNet(nn.Module):
    def __init__(self, args=None, embed_dim=512, img_size=224, drop_path=0.):
        super(ThermalRGBNet, self).__init__()

        self.rgb_backbone = pvt_v2_b3()
        self.thermal_backbone = pvt_v2_b3()
        if args:
            self.rgb_backbone.load_state_dict(torch.load(args.pretrained_model), strict=False)
            self.thermal_backbone.load_state_dict(torch.load(args.pretrained_model), strict=False)

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.mlp1_32 = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.t_mlp1_32 = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.mlp1_16 = nn.Sequential(
            nn.Linear(320, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.mlp1_8 = nn.Sequential(
            nn.Linear(128, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.mlp1_4 = nn.Sequential(
            nn.Linear(64, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.MSA = MultiScaleAttention(emb_dim=self.embed_dim, num_heads=8)
        self.Decoder_fuse_rgbt = transfmrerDecoder(depth=6, num_heads=8, embed_dim=self.embed_dim)
        self.count_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.count_pred = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 1),
        )
        self.pre_out = nn.Linear(self.embed_dim, 1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        rgb = inputs[0]
        thermal = inputs[1]
        B, _, H, W = rgb.shape
        N = (H // 32) * (W // 32)

        routs1, routs2 = self.rgb_backbone(rgb)
        touts1, touts2 = self.thermal_backbone(thermal)
        rgb_fea1_4, rgb_fea1_8, rgb_fea1_16, rgb_fea1_32 = routs2
        thermal_fea1_4, thermal_fea1_8, thermal_fea1_16, thermal_fea1_32 = touts2
        rgb_fea1_4 = self.mlp1_4(rgb_fea1_4)
        rgb_fea1_8 = self.mlp1_8(rgb_fea1_8)
        rgb_fea1_16 = self.mlp1_16(rgb_fea1_16)

        count_token = self.count_token.expand(B, -1, -1)
        out_fea = self.MSA(rgb_fea1_32, thermal_fea1_32, count_token)
        rgb_fea1_32 = out_fea[:, 0:N, :]
        t_token = out_fea[:, N:, :]
        rgb = torch.cat((rgb_fea1_4, rgb_fea1_8), dim=1)
        rgb = torch.cat((rgb, rgb_fea1_16), dim=1)
        rgb = torch.cat((rgb, rgb_fea1_32), dim=1)
        out_fea = self.Decoder_fuse_rgbt(t_token, rgb)
        count_token = out_fea[:, N:N+1, :]
        out_fea1 = out_fea[:, 0:N, :]
        count_pred = self.count_pred(count_token)
        out_pred = self.pre_out(out_fea1)

        out = out_pred.transpose(1, 2).reshape(B, 1, H // 32, W // 32)
        out = self.up4(out)
        mu = self.relu(out)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)

        return count_pred, torch.abs(mu), mu_normed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train ')
    # parser.add_argument('--pretrained_model',
    #                     default=r'', type=str,
    #                     help='load Pretrained model')
    args = parser.parse_args()
    a = torch.randn(1, 3, 224, 224)
    model = ThermalRGBNet(None)
    flops, params = profile(model, ([a,a],))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)

    c,d = model([a, a])
    # print(b.shape)
    print(c.shape)
    print(d.shape)