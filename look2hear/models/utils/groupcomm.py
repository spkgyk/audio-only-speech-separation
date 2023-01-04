import torch.nn as nn

from .gc3_basics import TAC, ProjRNN, split_feature, merge_feature
from .dptnet import DPTNet
from .dprnn import DPRNN
from .sudo_rm_rf import UConvBlock, GC_UConvBlock
from .tcn import TCN, GC_TCN

# GroupComm-RNN
class GC_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM", num_group=2, dropout=0, num_layers=1, bidirectional=False):
        super(GC_RNN, self).__init__()

        self.TAC = nn.ModuleList([])
        self.rnn = nn.ModuleList([])
        self.LN = nn.ModuleList([])

        self.num_layers = num_layers
        self.num_group = num_group

        for i in range(num_layers):
            self.TAC.append(TAC(input_size // num_group, hidden_size * 3 // num_group))
            self.rnn.append(ProjRNN(input_size // num_group, hidden_size // num_group, rnn_type, dropout, bidirectional))
            self.LN.append(nn.GroupNorm(1, input_size // num_group))

    def forward(self, input):
        # input shape: batch, dim, seq_len
        # split into groups
        batch_size, dim, seq_len = input.shape

        output = input.view(batch_size, self.num_group, -1, seq_len)
        for i in range(self.num_layers):
            # GroupComm via TAC
            output = self.TAC[i](output).transpose(2, 3).contiguous()  # B, G, L, N
            output = output.view(batch_size * self.num_group, seq_len, -1)  # B*G, L, dim

            # RNN
            rnn_output = self.rnn[i](output)
            norm_output = self.LN[i](rnn_output.transpose(1, 2))
            output = output + norm_output.transpose(1, 2)  # B*G, L, dim
            output = output.view(batch_size, self.num_group, seq_len, -1).transpose(2, 3).contiguous()  # B, G, dim, L

        output = output.view(batch_size, dim, seq_len)

        return output


# wrapper for dual-path models
class DP_Wrapper(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        num_group=16,
        layer=4,
        block_size=100,
        bidirectional=True,
        module="RNN",
        unfold=False,
    ):
        super(DP_Wrapper, self).__init__()

        assert module in [
            "DPRNN",
            "DPTNet",
        ]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.block_size = block_size
        self.num_spk = num_spk
        self.num_group = num_group
        self.unfold = unfold

        if module == "DPRNN":
            self.seq_model = DPRNN(
                self.input_dim,
                self.hidden_dim,
                self.output_dim * self.num_spk,
                num_layers=layer,
                num_group=self.num_group,
                bidirectional=bidirectional,
                unfold=self.unfold,
            )
        elif module == "DPTNet":
            self.seq_model = DPTNet(
                self.input_dim,
                self.hidden_dim,
                self.output_dim * self.num_spk,
                num_layers=layer,
                num_group=self.num_group,
                unfold=self.unfold,
            )

    def forward(self, input):

        batch_size = input.shape[0]

        # split the input into overlapped, longer segments
        input_blocks, input_rest = split_feature(input, self.block_size)  # B, N, L, K

        # pass to sequence modeling model
        output = self.seq_model(input_blocks).view(batch_size * self.num_spk, self.input_dim, self.block_size, -1)  # B, C, N, L, K

        # overlap-and-add of the outputs
        output = merge_feature(output, input_rest)
        output = output.view(batch_size, self.num_spk, self.output_dim, -1)  # B, C, K, T

        return output


class SudoRMRF_Wrapper(nn.Module):
    def __init__(self, out_channels, in_channels, upsampling_depth, layer, module, num_group=16):
        super(SudoRMRF_Wrapper, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.upsampling_depth = upsampling_depth
        self.layer = layer
        self.num_group = num_group

        if module == "GC_SudoRMRF":
            self.sudo_rmrf_layers = nn.Sequential(
                *[
                    GC_UConvBlock(
                        out_channels=out_channels, in_channels=in_channels, upsampling_depth=upsampling_depth, num_group=num_group
                    )
                    for _ in range(layer)
                ]
            )
        elif module == "SudoRMRF":
            self.sudo_rmrf_layers = nn.Sequential(
                *[UConvBlock(out_channels=out_channels, in_channels=in_channels, upsampling_depth=upsampling_depth) for _ in range(layer)]
            )

    def forward(self, input):
        return self.sudo_rmrf_layers(input)


class TCN_Wrapper(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer, stack, module, BN_dim=128, kernel=3, skip=True, dilated=True, num_group=2):
        super(TCN_Wrapper, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.skip = skip
        self.dilated = dilated

        if module == "TCN":
            self.tcn = TCN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                BN_dim=BN_dim,
                hidden_dim=self.hidden_dim,
                layer=self.layer,
                stack=self.stack,
                kernel=self.kernel,
                skip=self.skip,
                dilated=self.dilated,
            )
        elif module == "GC_TCN":
            self.tcn = GC_TCN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=self.hidden_dim,
                layer=self.layer,
                stack=self.stack,
                kernel=self.kernel,
                skip=self.skip,
                dilated=self.dilated,
                num_group=num_group,
            )

    def forward(self, input):
        return self.tcn(input)
