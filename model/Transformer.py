import torch.nn as nn
import torch
from model.IntmdSequential import IntermediateSequential
import torch.nn.functional as F
import numpy as np
import copy
from math import sqrt
import math

class EA(nn.Module):

    def __init__(self, in_channel, b=1, gama=2):

        super(EA, self).__init__()

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1

        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = x * inputs
        return outputs


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.ea = EA(7)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CalAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.sig = nn.Sigmoid()

    def cal_q(self, Q, K):
        B, H, L, E = K.shape
        _, _, S, _ = Q.shape

        sample_k = 1 * np.ceil(np.log(S)).astype('int').item()
        n_top = 2 * np.ceil(np.log(L)).astype('int').item()

        K_expand = K.unsqueeze(-3).expand(B, H, S, L, E)
        indx_sample = torch.randint(L, (S, sample_k))
        K_sample = K_expand[:, :, torch.arange(S).unsqueeze(1), indx_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L)
        M_v, M_top = M.topk(n_top, sorted=False)
        M_v = self.sig(M_v)

        return Q, M_top

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        _, M_top = self.cal_q(q, k)
        print('M_top',M_top)
        return M_top


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = IntermediateSequential(*layers)
        self.ca = CalAttention(dim, heads=heads, dropout_rate=attn_dropout_rate)

    def forward(self, x):
        m = self.net[3](x)
        m = self.ca(m)
        return self.net(x), m

