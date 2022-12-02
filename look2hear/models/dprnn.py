###
# Author: Kai Li
# Date: 2022-04-06 18:04:29
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-05-26 18:06:53
###
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from torch.autograd import Variable



class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(
            rnn_output.contiguous().view(-1, rnn_output.shape[2])
        ).view(output.shape)
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dropout=0,
        num_layers=1,
        bidirectional=True,
        full_causal=False,
    ):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.full_causal = full_causal
        self.bidirectional = bidirectional

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            if full_causal:
                self.row_rnn.append(
                    SingleRNN(
                        rnn_type, input_size, hidden_size, dropout, bidirectional=False
                    )
                )
                self.col_rnn.append(
                    SingleRNN(
                        rnn_type, input_size, hidden_size, dropout, bidirectional=False
                    )
                )
                self.row_norm.append(cLN(input_size, eps=1e-8))
                self.col_norm.append(cLN(input_size, eps=1e-8))
            else:
                self.row_rnn.append(
                    SingleRNN(
                        rnn_type, input_size, hidden_size, dropout, bidirectional=True
                    )
                )
                self.col_rnn.append(
                    SingleRNN(
                        rnn_type,
                        input_size,
                        hidden_size,
                        dropout,
                        bidirectional=bidirectional,
                    )
                )
                self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
                if bidirectional:
                    self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
                else:
                    self.col_norm.append(cLN(input_size, eps=1e-8))

        # output layer
        self.output = nn.Conv2d(input_size, output_size, 1)

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # dim1=chunk length, dim2=chunk number
        # import pdb; pdb.set_trace()
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            if self.full_causal:
                row_output = self.row_norm[i](
                    row_output.transpose(1, 2).contiguous()
                )  # B*dim2, H, dim1
                row_output = (
                    row_output.view(batch_size, dim2, -1, dim1)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                )  # B, N, dim1, dim2
            else:
                row_output = (
                    row_output.view(batch_size, dim2, dim1, -1)
                    .permute(0, 3, 2, 1)
                    .contiguous()
                )  # B, N, dim1, dim2
                row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            if self.full_causal or not self.bidirectional:
                col_output = self.col_norm[i](
                    col_output.transpose(1, 2).contiguous()
                )  # B*dim1, H, dim2
                col_output = (
                    col_output.view(batch_size, dim1, -1, dim2)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )  # B, N, dim1, dim2
            else:
                col_output = (
                    col_output.view(batch_size, dim1, dim2, -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )  # B, N, dim1, dim2
                col_output = self.col_norm[i](col_output)
            output = output + col_output

        output = self.output(output)

        return output


# base module for deep DPRNN
class DPRNN_base(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        bidirectional=True,
        model_type="DPRNN",
        group=4,
        rnn_type="LSTM",
        full_causal=False,
    ):
        super(DPRNN_base, self).__init__()

        assert model_type in ["DPRNN"], "model_type can only be 'DPRNN'"

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.model_type = model_type

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        self.DPRNN = getattr(sys.modules[__name__], model_type)(
            rnn_type,
            self.feature_dim,
            self.hidden_dim,
            self.output_dim * self.num_spk,
            num_layers=layer,
            bidirectional=bidirectional,
            full_causal=full_causal,
        )

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(
            input.type()
        )
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[:, :, :-segment_stride]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(2, 3)
            .contiguous()
            .view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass


def pad_segment(input, block_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    rest = block_size - (block_stride + seq_len % block_size) % block_size
    if rest > 0:
        pad = (
            Variable(torch.zeros(batch_size, dim, rest))
            .type(input.type())
            .to(input.device)
        )
        input = torch.cat([input, pad], 2)

    pad_aux = (
        Variable(torch.zeros(batch_size, dim, block_stride))
        .type(input.type())
        .to(input.device)
    )
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, block_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, block_size)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    block1 = (
        input[:, :, :-block_stride].contiguous().view(batch_size, dim, -1, block_size)
    )
    block2 = (
        input[:, :, block_stride:].contiguous().view(batch_size, dim, -1, block_size)
    )
    block = (
        torch.cat([block1, block2], 3)
        .view(batch_size, dim, -1, block_size)
        .transpose(2, 3)
    )

    return block.contiguous(), rest


def merge_feature(input, rest, return_all=False):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, block_size, _ = input.shape
    block_stride = block_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, block_size * 2)
    )  # B, N, K, L

    input1 = (
        input[:, :, :, :block_size]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, block_stride:]
    )
    input2 = (
        input[:, :, :, block_size:]
        .contiguous()
        .view(batch_size, dim, -1)[:, :, :-block_stride]
    )

    if rest > 0:
        input1 = input1[:, :, :-rest].contiguous()
        input2 = input2[:, :, :-rest].contiguous()
    output = input1 + input2

    if not return_all:
        return output.contiguous()  # B, N, T
    else:
        return output.contiguous(), input1, input2


class DPRNNSep(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(DPRNNSep, self).__init__(*args, **kwargs)

    def forward(self, input):

        batch_size = input.size(0)
        enc_feature = self.BN(input)

        # split the encoder output into overlapped, longer segments
        # this is for faster processing
        # first pad the segments accordingly
        enc_segments, enc_rest = self.split_feature(
            enc_feature, self.segment_size
        )  # B, N, L, K

        # pass to DPRNN
        output = self.DPRNN(enc_segments).view(
            batch_size * self.num_spk, self.output_dim, self.segment_size, -1
        )  # B, C, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)
        output = output.view(
            batch_size, self.num_spk, self.output_dim, -1
        )  # B, C, K, T

        return output


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
        return F.conv_transpose1d(
            x.reshape(view_as), self._filters, stride=self.stride, padding=self.padding
        ).view(x.shape[:-2] + (-1,))


class DPRNNTasNet(BaseModel):
    def __init__(
        self,
        feature_dim=128,
        hidden_dim=256,
        sample_rate=16000,
        win=32,
        layer=6,
        segment_size=32,
        context=1,
        num_spk=2,
        bidirectional=True,
    ):
        super(DPRNNTasNet, self).__init__(sample_rate=sample_rate)

        # parameters
        self.freq_win = sample_rate * win // 1000
        self.freq_stride = self.freq_win // 4
        self.encoder = Encoder(1, self.freq_win // 2 + 1, self.freq_win)

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_spk = num_spk

        self.eps = torch.finfo(torch.float32).eps

        # freq separator
        self.freq_norm = nn.GroupNorm(1, self.freq_win // 2 + 1, self.eps)
        self.freq_separator = DPRNNSep(
            self.freq_win // 2 + 1,
            self.feature_dim,
            self.hidden_dim,
            self.freq_win // 2 + 1,
            self.num_spk,
            layer=layer,
            segment_size=segment_size,
            full_causal=False,
            bidirectional=bidirectional,
        )

        # decoder
        self.decoder = Decoder(self.freq_win // 2 + 1, 2, self.freq_win)

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

    def forward(self, input_wav):
        # input shape: (B, T)
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        # pad input
        input_wav, rest = self.pad_input(input_wav, self.freq_win, self.freq_stride)
        input_wav = input_wav.unsqueeze(1)
        batch_size = input_wav.shape[0]

        # frequency-domain separation
        mixture_w = self.encoder(input_wav)

        # mask estimation
        mask = self.freq_separator(self.freq_norm(mixture_w))
        est_spec = mask * mixture_w.unsqueeze(1)
        est_spec = est_spec.view(batch_size, self.num_spk, mixture_w.shape[1], -1)

        decoder = self.decoder(est_spec)
        decoder = decoder[
            :,
            :,
            self.freq_win
            - self.freq_stride : -(rest + self.freq_win - self.freq_stride),
        ].contiguous()
        # estmite = self.pad_x_to_y(decoder, input_wav)
        if was_one_d:
            return decoder.squeeze(0)
        return decoder

    def pad_x_to_y(self, x, y, axis=-1):
        """Pad first argument to have same size as second argument

        Args:
            x (torch.Tensor): Tensor to be padded.
            y (torch.Tensor): Tensor to pad x to.
            axis (int): Axis to pad on.

        Returns:
            torch.Tensor, x padded to match y's shape.
        """
        if axis != -1:
            raise NotImplementedError
        inp_len = y.shape[axis]
        output_len = x.shape[axis]
        return nn.functional.pad(x, [0, inp_len - output_len])

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
