import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted
from model.variance import *
from model.STLDecomposer import *
from model.CMB import *
from layers.Attention import *


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.dropout = configs.dropout
        self.e_layers = configs.e_layers
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.var = Variance(1, self.dropout, configs.pred_len-1)

        scale_dim = 8
        self.mixer = MultTime2dMixer(time_step=configs.seq_len, channel=configs.c_out, scale_dim=configs.seq_len // 2, n=2)
        self.conv = nn.Conv1d(in_channels=configs.c_out, out_channels=configs.c_out, kernel_size=2, stride=2)
        self.output_expand = nn.Sequential(
            nn.Linear(int(2.5*self.seq_len), self.pred_len),
        )
        self.output_decrease = nn.Sequential(
            nn.Linear(int(self.seq_len), int(2.5*self.pred_len)),
        )
        self.cmblock = CMBlock(n_vars=configs.d_model)
        self.cmblockc_out = CMBlock(n_vars=configs.c_out)
        router_num = configs.router_num if hasattr(configs, 'router_num') and configs.router_num != 0 \
            else int(math.sqrt(configs.enc_in) + math.log2(configs.enc_in)) // 2
        if router_num % 2 != 0:
            router_num += 1
        self.RA = RotaryAttention(router_num = router_num, d_model=configs.d_model, n_heads=configs.n_heads)
        self.qkv = QKV(d_model=configs.d_model, n_heads=configs.n_heads)
        self.period = configs.period
        self.trend_kernel = configs.trend_kernel

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, L, N = x_enc.shape  # B L N
        x_enc_stl = x_enc.reshape(B * L, N)
        season, trend = run_stl(x_enc_stl, self.period, self.trend_kernel)
        season = season.reshape(B, L, N)
        trend = trend.reshape(B, L, N)
        x_enc = season

        x = trend.permute(0, 2, 1)  # x:[1026,5,16]
        x = self.conv(x)  # x:[1026,5,8]
        x = x.permute(0, 2, 1)  # x:[1026,8,5]
        y = self.mixer(trend, x)  # y:[1026,40,5]
        y = y.permute(0, 2, 1)
        y = self.output_expand(y)
        y = y.permute(0, 2, 1)

        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)


        x_enc2 = x_enc.transpose(2, 1)
        x_enc2 = x_enc2.reshape(B*N, L, 1)

        enc_rot,_ = self.RA(enc_out)
        enc_rot =self.cmblock(enc_out,enc_rot)

        dec_out = self.projector(enc_rot).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out +y

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]