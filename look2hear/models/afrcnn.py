'''
Author: Kai Li
Date: 2020-08-09 17:32:53
LastEditTime: 2021-05-21 22:54:23
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .base_model import BaseModel

class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.
        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class Blocks(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=5,
                                               stride=2,
                                               groups=in_channels, d=1))
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i == j:
                    fuse_layer.append(None)
                elif j-i == 1:
                    fuse_layer.append(None)
                elif i-j == 1:
                    fuse_layer.append(DilatedConvNorm(in_channels, in_channels,
                                                      kSize=5,
                                                      stride=2,
                                                      groups=in_channels, d=1))
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == upsampling_depth-1:
                self.concat_layer.append(ConvNormAct(
                    in_channels*2, in_channels, 1, 1))
            else:
                self.concat_layer.append(ConvNormAct(
                    in_channels*3, in_channels, 1, 1))

        self.last_layer = nn.Sequential(
            ConvNormAct(in_channels*upsampling_depth, in_channels, 1, 1)
        )
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            y = torch.cat((self.fuse_layers[i][0](output[i-1]) if i-1 >= 0 else torch.Tensor().to(output1.device),
                           output[i],
                           F.interpolate(output[i+1], size=wav_length, mode='nearest') if i+1 < self.depth else torch.Tensor().to(output1.device)), dim=1)
            x_fuse.append(self.concat_layer[i](y))

        wav_length = output[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(
                x_fuse[i], size=wav_length, mode='nearest')

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual
        #return expanded

class Recurrent(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 _iter=4):
        super().__init__()
        self.blocks = Blocks(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        #self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.blocks(x)
            else:
                #m = self.attention(mixture, x)
                x = self.blocks(self.concat_block(mixture+x))
        return x


class AFRCNN(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=16000,
    ):
        super(AFRCNN, self).__init__(sample_rate=sample_rate)
        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(
            self.enc_kernel_size // 4 * 4 ** self.upsampling_depth
        ) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis * num_sources,
            out_channels=num_sources,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

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

    # Forward pass
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

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        # Front end
        x = self.encoder(x.unsqueeze(1))

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = estimated_waveforms[
            :,
            :,
            self.enc_kernel_size
            - self.enc_kernel_size
            // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        ].contiguous()
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args