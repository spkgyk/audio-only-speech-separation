"""!
@author Yi Luo (oulyluo)
@copyright Tencent AI Lab
"""

import torch
import torch.nn as nn

from numpy import floor
from .utils import BaseModel


class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0, bidirectional=True):
        super(ResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps

        self.norm = nn.GroupNorm(1, input_size, self.eps)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(bidirectional) + 1), input_size)

    def forward(self, input):
        # input shape: batch, dim, seq

        rnn_output, _ = self.rnn(self.dropout(self.norm(input)).transpose(1, 2).contiguous())
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(input.shape[0], input.shape[2], input.shape[1])

        return input + rnn_output.transpose(1, 2).contiguous()


class BSNet(nn.Module):
    def __init__(self, in_channel, nband=7, num_layer=1, dropout=0.0, bi_comm=True):
        super(BSNet, self).__init__()

        self.nband = nband
        self.feature_dim = in_channel // nband

        self.band_rnn = []
        for _ in range(num_layer):
            self.band_rnn.append(ResRNN(self.feature_dim, self.feature_dim * 2, dropout))
        self.band_rnn = nn.Sequential(*self.band_rnn)
        self.band_comm = ResRNN(self.feature_dim, self.feature_dim * 2, dropout, bi_comm)

    def forward(self, input):
        # input shape: B, nband*N, T
        B, N, T = input.shape

        band_output = self.band_rnn(input.view(B * self.nband, self.feature_dim, -1)).view(B, self.nband, -1, T)

        # band comm
        band_output = band_output.permute(0, 3, 2, 1).contiguous().view(B * T, -1, self.nband)
        output = self.band_comm(band_output).view(B, T, -1, self.nband).permute(0, 3, 2, 1).contiguous()

        return output.view(B, N, T)


class BSRNN(BaseModel):
    def __init__(
        self,
        win=2048,
        stride=512,
        feature_dim=128,
        num_layer=1,
        num_repeat=12,
        context=0,
        dropout=0.0,
        bi_comm=True,
        sample_rate=44100,
    ):
        super(BSRNN, self).__init__(sample_rate=sample_rate)

        self.sr = sample_rate
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.context = context
        self.ratio = context * 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps

        # split v7
        # 0-1k (100 hop), 1k-4k (250 hop), 4k-8k (500 hop), 8k-16k (1k hop), 16k-20k (2k hop), 20k-inf
        bandwidth_100 = int(floor(100 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_250 = int(floor(250 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_500 = int(floor(500 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_1k = int(floor(1000 / (sample_rate / 2.0) * self.enc_dim))
        bandwidth_2k = int(floor(2000 / (sample_rate / 2.0) * self.enc_dim))
        self.band_width = [bandwidth_100] * 10
        self.band_width += [bandwidth_250] * 12
        self.band_width += [bandwidth_500] * 8
        self.band_width += [bandwidth_1k] * 8
        self.band_width += [bandwidth_2k] * 2
        self.band_width.append(self.enc_dim - sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(
                nn.Sequential(nn.GroupNorm(1, self.band_width[i] * 2, self.eps), nn.Conv1d(self.band_width[i] * 2, self.feature_dim, 1))
            )

        self.separator = []
        for i in range(num_repeat):
            self.separator.append(BSNet(self.nband * self.feature_dim, self.nband, num_layer, dropout, bi_comm))
        self.separator = nn.Sequential(*self.separator)

        self.mask = nn.ModuleList([])
        for i in range(self.nband):
            self.mask.append(
                nn.Sequential(
                    nn.GroupNorm(1, self.feature_dim, torch.finfo(torch.float32).eps),
                    nn.Conv1d(self.feature_dim, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.feature_dim * 4, 1),
                    nn.Tanh(),
                    nn.Conv1d(self.feature_dim * 4, self.band_width[i] * self.ratio * 4, 1),
                )
            )

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
        pad_aux = torch.zeros(batch_size, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    def forward(self, input):
        # input shape: (B, C, T)

        batch_size, nch, nsample = input.shape
        input = input.view(batch_size * nch, -1)

        # frequency-domain separation
        spec = torch.stft(
            input,
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            return_complex=True,
        )

        # get a context
        prev_context = []
        post_context = []
        zero_pad = torch.zeros_like(spec)
        for i in range(self.context):
            this_prev_context = torch.cat([zero_pad[:, : i + 1], spec[:, : -1 - i]], 1)
            this_post_context = torch.cat([spec[:, i + 1 :], zero_pad[:, : i + 1]], 1)
            prev_context.append(this_prev_context)
            post_context.append(this_post_context)
        mixture_context = torch.stack(prev_context + [spec] + post_context, 1)  # B*nch, K, F, T

        # concat real and imag, split to subbands
        spec_RI = torch.stack([spec.real, spec.imag], 1)  # B*nch, 2, F, T
        subband_spec = []
        subband_spec_context = []
        subband_power = []
        band_idx = 0
        for i in range(len(self.band_width)):
            subband_spec.append(spec_RI[:, :, band_idx : band_idx + self.band_width[i]].contiguous())
            subband_spec_context.append(mixture_context[:, :, band_idx : band_idx + self.band_width[i]])  # B*nch, K, BW, T
            subband_power.append((subband_spec_context[-1].abs().pow(2).mean(1).mean(1) + self.eps).sqrt())  # B*nch, T
            band_idx += self.band_width[i]

        # normalization and bottleneck
        subband_feature = []
        for i in range(len(self.band_width)):
            subband_feature.append(self.BN[i](subband_spec[i].view(batch_size * nch, self.band_width[i] * 2, -1)))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        # separator
        sep_output = self.separator(subband_feature.view(batch_size * nch, self.nband * self.feature_dim, -1))  # B, nband*N, T
        sep_output = sep_output.view(batch_size * nch, self.nband, self.feature_dim, -1)

        sep_subband_spec = []
        for i in range(len(self.band_width)):
            this_output = self.mask[i](sep_output[:, i]).view(batch_size * nch, 2, 2, self.ratio, self.band_width[i], -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # B*nch, 2, K, BW, T
            this_mask_real = this_mask[:, 0]  # B*nch, K, BW, T
            this_mask_imag = this_mask[:, 1]  # B*nch, K, BW, T
            est_spec_real = (subband_spec_context[i].real * this_mask_real).mean(1) - (subband_spec_context[i].imag * this_mask_imag).mean(
                1
            )  # B*nch, BW, T
            est_spec_imag = (subband_spec_context[i].real * this_mask_imag).mean(1) + (subband_spec_context[i].imag * this_mask_real).mean(
                1
            )  # B*nch, BW, T
            sep_subband_spec.append(torch.complex(est_spec_real, est_spec_imag))
        est_spec = torch.cat(sep_subband_spec, 1)  # B*nch, F, T

        output = torch.istft(
            est_spec.view(batch_size * nch, self.enc_dim, -1),
            n_fft=self.win,
            hop_length=self.stride,
            window=torch.hann_window(self.win).to(input.device).type(input.type()),
            length=nsample,
        )

        output = output.view(batch_size, nch, -1)

        return output

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
