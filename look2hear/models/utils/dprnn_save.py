import torch.nn as nn

from .gc3_basics import ProjRNN, TAC

# dual-path RNN
class DPRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0, num_layers=1, bidirectional=True, unfold=False):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.factor = int(bidirectional) + 1
        self.unfold = unfold
        self.num_layers = num_layers

        # dual-path RNN
        if self.unfold:
            self.row_rnn = ProjRNN(input_size, hidden_size, "LSTM", dropout, bidirectional=True)
            self.col_rnn = ProjRNN(input_size, hidden_size, "LSTM", dropout, bidirectional=bidirectional)
            self.row_norm = nn.GroupNorm(1, input_size, eps=1e-8)
            self.col_norm = nn.GroupNorm(1, input_size, eps=1e-8)
        else:
            self.row_rnn = nn.ModuleList([])
            self.col_rnn = nn.ModuleList([])
            self.row_norm = nn.ModuleList([])
            self.col_norm = nn.ModuleList([])
            for i in range(self.num_layers):
                self.row_rnn.append(ProjRNN(input_size, hidden_size, "LSTM", dropout, bidirectional=True))
                self.col_rnn.append(ProjRNN(input_size, hidden_size, "LSTM", dropout, bidirectional=bidirectional))
                self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Conv2d(input_size, output_size, 1)

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2

        batch_size, _, dim1, dim2 = input.shape
        output = input

        for i in range(self.num_layers):
            if self.unfold:
                row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
                row_output = self.row_rnn(row_input)  # B*dim2, dim1, H
                row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
                row_output = self.row_norm(row_output)
                output = output + row_output

                col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
                col_output = self.col_rnn(col_input)  # B*dim1, dim2, H
                col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
                col_output = self.col_norm(col_output)
                output = self.concat_block(output + col_output)
            else:
                row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
                row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
                row_output = row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
                row_output = self.row_norm[i](row_output)
                output = output + row_output

                col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
                col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
                col_output = col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
                col_output = self.col_norm[i](col_output)
                output = output + col_output

        output = self.output(output)

        return output


# GroupComm-DPRNN
class GC_DPRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_group=2, dropout=0, num_layers=1, bidirectional=True, unfold=False):
        super(GC_DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_group = num_group
        self.num_spk = output_size // input_size
        self.factor = int(bidirectional) + 1
        self.unfold = unfold
        self.num_layers = num_layers

        if self.unfold:
            # dual-path RNN with TAC
            self.TAC = TAC(input_size // num_group, hidden_size * 3 // num_group)
            self.row_rnn = ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=True)
            self.col_rnn = ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=bidirectional)
            self.row_norm = nn.GroupNorm(1, input_size // num_group, eps=1e-8)
            self.col_norm = nn.GroupNorm(1, input_size // num_group, eps=1e-8)
        else:
            # dual-path RNN with TAC
            self.TAC = nn.ModuleList([])
            self.row_rnn = nn.ModuleList([])
            self.col_rnn = nn.ModuleList([])
            self.row_norm = nn.ModuleList([])
            self.col_norm = nn.ModuleList([])
            for i in range(self.num_layers):
                self.TAC.append(TAC(input_size // num_group, hidden_size * 3 // num_group))
                self.row_rnn.append(ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=True))
                self.col_rnn.append(
                    ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=bidirectional)
                )
                self.row_norm.append(nn.GroupNorm(1, input_size // num_group, eps=1e-8))
                self.col_norm.append(nn.GroupNorm(1, input_size // num_group, eps=1e-8))

        self.output = nn.Conv2d(input_size // num_group, output_size // num_group, 1)

    def forward(self, input):
        # input shape: batch, N, dim1, dim2

        batch_size, N, dim1, dim2 = input.shape
        output = input.view(batch_size, self.num_group, -1, dim1, dim2)

        for i in range(self.num_layers):
            if self.unfold:
                # GroupComm
                output = self.TAC(output.view(batch_size, self.num_group, -1, dim1 * dim2))  # B, G, N/G, dim1*dim2
                output = output.view(batch_size * self.num_group, -1, dim1, dim2)  # B*G, N/G, dim1, dim2

                # intra-block
                row_input = (
                    output.permute(0, 3, 2, 1).contiguous().view(batch_size * self.num_group * dim2, dim1, -1)
                )  # B*G*dim2, dim1, N/G
                row_output = self.row_rnn(row_input)  # B*G*dim2, dim1, N/G
                row_output = row_output.view(batch_size * self.num_group, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
                # ^    B*G, N/G, dim1, dim2
                row_output = self.row_norm(row_output.view(batch_size * self.num_group, -1, dim1, dim2)).view(output.shape)
                # ^    B*G, N/G, dim1, dim2
                output = output + row_output  # B*G, N/G, dim1, dim2

                # inter-block
                col_input = (
                    output.permute(0, 2, 3, 1).contiguous().view(batch_size * self.num_group * dim1, dim2, -1)
                )  # B*G*dim1, dim2, N/G
                col_output = self.col_rnn(col_input)  # B*G*dim1, dim2, N/G
                col_output = col_output.view(batch_size * self.num_group, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
                # ^    B*G, N/G, dim1, dim2
                col_output = self.col_norm(col_output.view(batch_size * self.num_group, -1, dim1, dim2)).view(output.shape)
                # ^    B*G, N/G, dim1, dim2
                output = output + col_output  # B*G, N/G, dim1, dim2
            else:
                # GroupComm
                output = self.TAC[i](output.view(batch_size, self.num_group, -1, dim1 * dim2))  # B, G, N/G, dim1*dim2
                output = output.view(batch_size * self.num_group, -1, dim1, dim2)  # B*G, N/G, dim1, dim2

                # intra-block
                row_input = (
                    output.permute(0, 3, 2, 1).contiguous().view(batch_size * self.num_group * dim2, dim1, -1)
                )  # B*G*dim2, dim1, N/G
                row_output = self.row_rnn[i](row_input)  # B*G*dim2, dim1, N/G
                row_output = row_output.view(batch_size * self.num_group, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
                # ^    B*G, N/G, dim1, dim2
                row_output = self.row_norm[i](row_output.view(batch_size * self.num_group, -1, dim1, dim2)).view(output.shape)
                # ^    B*G, N/G, dim1, dim2
                output = output + row_output  # B*G, N/G, dim1, dim2

                # inter-block
                col_input = (
                    output.permute(0, 2, 3, 1).contiguous().view(batch_size * self.num_group * dim1, dim2, -1)
                )  # B*G*dim1, dim2, N/G
                col_output = self.col_rnn[i](col_input)  # B*G*dim1, dim2, N/G
                col_output = col_output.view(batch_size * self.num_group, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
                # ^    B*G, N/G, dim1, dim2
                col_output = self.col_norm[i](col_output.view(batch_size * self.num_group, -1, dim1, dim2)).view(output.shape)
                # ^    B*G, N/G, dim1, dim2
                output = output + col_output  # B*G, N/G, dim1, dim2

        output = output.view(batch_size * self.num_group, -1, dim1, dim2)
        output = self.output(output).view(batch_size, self.num_group, self.num_spk, -1, dim1, dim2)
        output = output.transpose(1, 2).contiguous()

        return output
