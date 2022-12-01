###
# Author: Kai Li
# Date: 2021-09-06 12:37:50
# LastEditors: Please set LastEditors
# LastEditTime: 2021-12-13 19:31:51
###
import torch
import inspect
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from typing import Optional
import math
import torch.nn.functional as F
from torch.nn.functional import unfold, fold
from ..layers import normalizations, activations
from .base_model import BaseModel

def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if `fn` can be called with `name` as a keyword
            argument.

    Returns:
        bool: whether `fn` accepts a `name` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class PositionalEncoding(nn.Module):

    def __init__(self, in_channels, max_length):
        super().__init__()
        pe = torch.zeros(max_length, in_channels)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_channels, 2).float()
                             * (-math.log(10000.0)/in_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class GlobalAttnLayer(nn.Module):
    def __init__(self, in_channels, n_head, dropout, is_casual):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_channels, n_head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.is_casual = is_casual

    def forward(self, x, spk_inp=None):
        attns = None
        if spk_inp is not None:
            output, _ = self.attn(x, spk_inp, spk_inp)
        else: 
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
            if mask[mask.size(0)-1, 0] != 0:
                mask = mask.t()
            output, _ = self.attn(x, x, x,
                attn_mask=mask if self.is_casual else None)
        output = self.norm(output + self.dropout(output))
        return output, attns

class DeepGlobalAttnLayer(nn.Module):
    def __init__(self, in_channels, n_head, dropout, is_casual):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels, 500)
        self.attn_in_norm = nn.LayerNorm(in_channels)
        self.attn_layer = nn.ModuleList([GlobalAttnLayer(in_channels, n_head, dropout, is_casual)
            for _ in range(1)])
        
    def forward(self, x, spk_inp=None):
        output = self.pos_enc(self.attn_in_norm(x))
        for block in self.attn_layer:
            output, attns = block(output, spk_inp=spk_inp)
        return output, attns

class SingleRNN(nn.Module):
    """Module for a RNN block.
    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.
    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """
    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 n_layers=1,
                 dropout=0,
                 bidirectional=False):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output

class SandglassetBlock(nn.Module):
    def __init__(
        self,
        in_chan,
        hid_size,
        n_head=8,
        norm_type="gLN",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        block_i=2,
        model_n_block=6,
        chunk_size=64
    ):
        super(SandglassetBlock, self).__init__()
        self.intra_RNN = SingleRNN(
            rnn_type,
            in_chan,
            hid_size,
            num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.inter_RNN = DeepGlobalAttnLayer(in_chan, n_head, dropout, False)

        self.intra_linear = nn.Linear(self.intra_RNN.output_size, in_chan)
        self.intra_norm = nn.GroupNorm(1, in_chan)

        # self.inter_linear = nn.Linear(in_chan, in_chan)
        self.inter_norm = nn.GroupNorm(1, in_chan)

        if block_i < model_n_block//2:
            kernel_size = 4 ** (block_i)
        else:
            kernel_size = 4 ** (model_n_block-block_i-1)
        self.downsampler = nn.AvgPool1d(kernel_size, stride=kernel_size)
        self.upsampler = nn.Upsample(size=chunk_size, mode='linear', align_corners=True)

    def forward(self, x, skip_connect=None, spk_inp=None):
        """ Input shape : [batch, feats, chunk_size, num_chunks] """
        # B, N, K, L = x.size()
        # output = x  # for skip connection
        B, D, K, S = x.size()

        # Intra-chunk processing
        local_input = x.transpose(1, 3).reshape(B * S, K, D)
        local_output = self.intra_RNN(local_input)
        local_output = self.intra_linear(local_output)
        local_output = local_output.view(B, S, K, D).transpose(1, 3).contiguous()
        local_output = self.intra_norm(local_output)
        x = x + local_output

        # Inter-chunk processing
        global_input = x.permute(3, 0, 1, 2).contiguous().view(S*B, D, K)
        # downsample
        global_input = self.downsampler(global_input)
        Q = global_input.size(-1)
        global_input = global_input.transpose(1, 2).reshape(S, B*Q, D)
        if skip_connect is not None:
            global_input = global_input + skip_connect
        global_output, attns = self.inter_RNN(global_input)

        skip_connect_output = global_output.clone()
        # [S, B*Q, D] -> [B, D*S, Q]
        global_output = global_output.view(S, B, Q, D).permute(1, 3, 0, 2).reshape(B, D*S, Q)
        # [B, D*S, Q] -> [B, D, K, S]
        global_output = self.upsampler(global_output).view(B, D, S, K).transpose(2, 3).contiguous()
        global_output = self.inter_norm(global_output).view(B, D, K, S)
        x = x + global_output
        return x, skip_connect_output


class Decoder(nn.Module):

    def __init__(self, in_chan, kernel_size):
        super().__init__()
        self.basis_lin = nn.Linear(in_chan, kernel_size, bias=False)
        self.kernel_size = kernel_size

    def forward(self, est_frames):
        """
        Args:
            est_frames: [B, C, I, D]
            B = batch size
            C = number of speakers
            I = number of frames
            M = frame dimension
        Returns:
            est_sigs: [B, C, T]
            T = number of samples
        """
        # Map frame dims back to window length, [B, C, D, I] -> [B, C, I, M]
        est_frames = self.basis_lin(est_frames.transpose(2, 3))
        est_sigs = self.overlap_and_add(est_frames,  self.kernel_size // 2)
        return est_sigs

    def merge_feature(self, input):
        B, C, I, M = input.shape
        input = input.view(B, C, -1, M * 2)

        input1 = input[:, :, :, :M].contiguous().view(B, C, -1)[:, :, M//2:]
        input2 = input[:, :, :, M:].contiguous().view(B, C, -1)[:, :, :-M//2]
        output = input1 + input2

        return output.contiguous()

    ## Below are the methods to be called by the main methods 
    def overlap_and_add(self, signal, frame_step):
        import math
        outer_dimensions = signal.size()[:-2]
        frames, frame_length = signal.size()[-2:]
        subframe_length = math.gcd(frame_length, frame_step)
        subframe_step = frame_step // subframe_length
        subframes_per_frame = frame_length // subframe_length
        output_size = frame_step * (frames - 1) + frame_length
        output_subframes = output_size // subframe_length
        subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
        frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
        frame = signal.new_tensor(frame).long().to(signal.device)
        frame = frame.contiguous().view(-1)
        result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length).to(signal.device)
        result.index_add_(-2, frame.to(signal.device), subframe_signal)
        result = result.view(*outer_dimensions, -1)
        return result


class Sandglasset2(BaseModel):
    def __init__(self,
                 n_feats=64,
                 n_src=2,
                 out_chan=64,
                 bn_chan=128,
                 hid_size=128,
                 chunk_size=250,
                 hop_size=125,
                 n_repeats=6,
                 n_head=8,
                 norm_type='gLN',
                 mask_act='sigmoid',
                 bidirectional=True,
                 rnn_type='LSTM',
                 num_layers=1,
                 dropout=0,
                 # encoder/decoder
                 kernel_size=2,
                 sr=16000):
        super(Sandglasset2, self).__init__(sample_rate=sr)
        # encoder part
        self.encoder = nn.Conv1d(1, n_feats, kernel_size=kernel_size, stride=kernel_size//2, bias=False)
        self.enc_LN = nn.GroupNorm(1, n_feats, eps=1e-8)
        self.bottleneck = nn.Conv1d(n_feats, bn_chan, 1, bias=False)
        self.seg_norm = nn.GroupNorm(1, bn_chan, eps=1e-8)
        self.out_norm = nn.GroupNorm(1, n_feats, eps=1e-8)

        # separation part
        sep = nn.ModuleList([])
        for x in range(n_repeats):
            sep.append(SandglassetBlock(
                bn_chan,
                hid_size,
                n_head,
                norm_type,
                bidirectional,
                rnn_type,
                num_layers,
                dropout,
                x,
                n_repeats,
                chunk_size
            ))
        self.sep_net = sep

        # Masking generat part
        sep_out_conv = nn.Conv2d(bn_chan, n_src*n_feats, 1)
        self.first_out = nn.Sequential(nn.PReLU(), sep_out_conv, nn.Softplus())

        self.out_norm = nn.GroupNorm(1, n_feats, eps=1e-8)

        self.decoder = Decoder(n_feats, kernel_size)
        # super-parameters
        self.bn_chan = bn_chan
        self.n_src = n_src
        self.out_chan = n_feats
        self.chunk_size = chunk_size
        self.hop_size =hop_size
        self.kernel_size = kernel_size

    def forward(self, input_wav):
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav

        sig_lens = input_wav.shape[-1]
        input_wav = self.normalize_signal(input_wav, sig_lens, 5)
        input_wav, rest, sig_lens = self.pad_zeros(input_wav, sig_lens)
        # encoder part
        mixture_w = F.relu(self.encoder(input_wav.unsqueeze(1)))
        mixture_w = self.enc_LN(mixture_w)
        output = self.bottleneck(mixture_w)
        B, D, I = output.size()
        output, ori_len = self.split_feature(output, self.chunk_size)
        output = self.seg_norm(F.relu(output))
        S = output.size(-1)
        K = self.chunk_size
        C = self.n_src

        # separation part
        skip_connect_outputs = []
        for i, block in enumerate(self.sep_net):
            if i < len(self.sep_net)//2:
                output, skip_connect_output = block(output)
                skip_connect_outputs.append(skip_connect_output)
            else:
                output, _ = block(output, skip_connect=skip_connect_outputs[-1])
                del skip_connect_outputs[-1]

        # Masking part
        output = self.first_out(output)
        est_masks = self.merge_feature(output.view(B * C, -1, K, S), ori_len).view(B*C, -1, I)
        # Result
        est_masks = self.out_norm(F.relu(est_masks)).view(B, C, -1, I)

        # Decoder
        masked_mix_w = est_masks * mixture_w.unsqueeze(1)
        estmite = self.decoder(masked_mix_w)[:, :, self.hop_size:-(rest+self.hop_size)]
        if was_one_d:
            return estmite.squeeze(0)
        return estmite
    
    def normalize_signal(self, sig, sig_lens, snr=0.):
        sig = sig-sig.sum(-1, keepdim=True)/sig_lens
        sig = sig/(torch.max(torch.abs(sig), -1, keepdim=True)[0]+1e-12)
        if snr == 0:
            return sig
        return sig/(10**(snr/20.))
    
    def pad_zeros(self, signals, sig_lens):
        B, T = signals.shape
        win_len = self.kernel_size
        self.hop_size = win_len // 2
        rest = win_len - (self.hop_size + T % win_len) % win_len
        if rest > 0:
            pad = torch.zeros(B, rest).type(signals.type()).to(signals.device)
            signals = torch.cat([signals, pad], 1)
        pad_aux = torch.zeros(B, self.hop_size).type(signals.type()).to(signals.device)
        signals = torch.cat([pad_aux, signals, pad_aux], 1)
        sig_lens += 2 * self.hop_size
        return signals, rest, sig_lens
    
    def split_feature(self, x, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, D, T)
        batch, dim, frames = x.size()
        ori_len = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(segment_size, 1),
            padding=(segment_size, 0),
            stride=(segment_size // 2, 1),
        )
        return unfolded.reshape(batch, dim, segment_size, -1), ori_len
    
    def merge_feature(self, x, ori_len):
        # merge the splitted features into full utterance
        # input is the features: (B, D, K, S)

        batch, dim, segment_size, n_segments = x.size()
        to_unfold = x.reshape(batch, dim * segment_size, n_segments)
        x = torch.nn.functional.fold(
            to_unfold, (ori_len, 1),
            kernel_size=(segment_size, 1),
            padding=(segment_size, 0),
            stride=(segment_size // 2, 1),
        ) / 2.

        return x.reshape(batch, dim, ori_len)

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