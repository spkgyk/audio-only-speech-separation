###
# Author: Kai Li
# Date: 2021-09-06 12:37:50
# LastEditors: Please set LastEditors
# LastEditTime: 2022-05-26 21:07:13
###
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import normalizations, BaseModel


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class Conv1D_Block(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=512,
        kernel_size=3,
        dilation=1,
        norm_type="gLN",
    ):
        super(Conv1D_Block, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = normalizations.get(norm_type)(out_channels)
        if norm_type == "gLN":
            self.padding = (dilation * (kernel_size - 1)) // 2
        else:
            self.padding = dilation * (kernel_size - 1)
        self.dwconv = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=self.padding,
            groups=out_channels,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = normalizations.get(norm_type)(out_channels)
        self.sconv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.norm_type = norm_type

    def forward(self, x):
        w = self.conv1x1(x)
        w = self.norm1(self.prelu1(w))
        w = self.dwconv(w)
        if self.norm_type == "cLN":
            w = w[:, :, : -self.padding]
        w = self.norm2(self.prelu2(w))
        w = self.sconv(w)
        x = x + w
        return x


class TCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type="gLN", X=8):
        super(TCN, self).__init__()
        seq = []
        for i in range(X):
            seq.append(
                Conv1D_Block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                    dilation=2**i,
                )
            )
        self.tcn = nn.Sequential(*seq)

    def forward(self, x):
        return self.tcn(x)


class Separation(nn.Module):
    def __init__(
        self,
        in_channels=128,
        out_channels=512,
        kernel_size=3,
        norm_type="gLN",
        X=8,
        R=3,
    ):
        super(Separation, self).__init__()
        s = [
            TCN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                X=X,
            )
            for i in range(R)
        ]
        self.sep = nn.Sequential(*s)

    def forward(self, x):
        return self.sep(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super(Encoder, self).__init__()
        self._filters = nn.Parameter(torch.ones(out_channel, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)
        self.stride = kernel_size // 4
        self.filters = out_channel
        self.padding = padding

    def forward(self, x):
        return F.conv1d(x, self._filters, stride=self.stride, padding=self.padding)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0, out_padding=0):
        super(Decoder, self).__init__()
        self._filters = nn.Parameter(torch.ones(in_channel, 1, kernel_size))
        for p in self.parameters():
            nn.init.xavier_normal_(p)
        self.stride = kernel_size // 4
        self.filters = in_channel
        self.padding = padding

    def forward(self, x):
        view_as = (-1,) + x.shape[-2:]
        return F.conv_transpose1d(x.reshape(view_as), self._filters, stride=self.stride, padding=self.padding).view(x.shape[:-2] + (-1,))


class ConvTasNet(BaseModel):
    def __init__(
        self,
        N=512,
        L=16,
        B=128,
        H=512,
        P=3,
        X=8,
        R=3,
        norm="gLN",
        num_spks=2,
        activate="relu",
        causal=False,
        sample_rate=16000,
    ):
        super(ConvTasNet, self).__init__(sample_rate=sample_rate)
        # -----------------------model-----------------------
        self.encoder = Encoder(1, N, L)
        self.bottleneck = nn.Sequential(
            normalizations.get("cLN")(N) if causal else normalizations.get("gLN")(N),
            nn.Conv1d(N, B, 1),
        )
        self.separation = Separation(B, H, P, norm, X, R)
        self.decoder = Decoder(N, 1, L)
        self.mask = nn.Conv1d(B, N * num_spks, 1, 1)
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax,
        }
        if activate not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}", format(activate))
        self.non_linear = supported_nonlinear[activate]
        self.num_spks = num_spks
        self.win = L
        self.stride = L // 2
        self.model_name = "ConvTasNet"

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, x):
        x, rest = self.pad_input(x, self.win, self.stride)
        x = self.encoder(x.unsqueeze(1))
        batch, ndim, times = x.shape
        w = self.bottleneck(x)
        w = self.separation(w)
        m = self.mask(w)
        m = self.non_linear(m)
        d = x.unsqueeze(1) * m.reshape(batch, self.num_spks, -1, times)  # [B, N*n_src, T]
        # s = torch.cat([self.decoder(d[i]) for i in range(self.num_spks)],dim=1).view(batch*self.num_spks, -1)
        s = self.decoder(d.reshape(batch * self.num_spks, -1, times))
        separate_output = s[:, self.win - self.stride : -(rest + self.win - self.stride)].contiguous().view(batch, self.num_spks, -1)
        return separate_output

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
