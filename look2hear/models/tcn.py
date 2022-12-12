import torch.nn as nn

from .basics import TAC

# TCN
class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation, groups=hidden_channel, padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False, dilated=True):
        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal))
                else:
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # output layer

        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

        self.skip = skip

    def forward(self, input):

        # input shape: (B, N, L)

        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output


# GroupComm-TCN
class GC_TCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer, stack, kernel=3, skip=True, causal=False, dilated=True, num_group=2):
        super(GC_TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        # TCN
        self.receptive_field = 0
        self.dilated = dilated
        self.num_group = num_group

        self.TAC = nn.ModuleList([])
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            input_dim // num_group,
                            hidden_dim // num_group,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            input_dim // num_group, hidden_dim // num_group, kernel, dilation=1, padding=1, skip=skip, causal=causal
                        )
                    )
                self.TAC.append(TAC(input_dim // num_group, hidden_dim * 3 // num_group))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # output layer

        self.output = nn.Conv1d(input_dim // num_group, output_dim // num_group, 1)

        self.skip = skip

    def forward(self, input):

        # input shape: (B, N, L)
        batch_size, N, L = input.shape
        output = input.view(batch_size, self.num_group, -1, L)  # B, G, N/G, L

        # pass to TCN
        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)  # B, G, N/G, L
                output = output.view(batch_size * self.num_group, -1, L)  # B*G, N/G, L
                residual, skip = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)  # B, G, N/G, L
                skip_connection = skip_connection + skip  # B*G, N/G, L
        else:
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)  # B, G, N/G, L
                output = output.view(batch_size * self.num_group, -1, L)  # B*G, N/G, L
                residual = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)  # B, G, N/G, L

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output.view(batch_size * self.num_group, -1, L))

        output = output.view(batch_size, -1, L)  # B, N, L

        return output
