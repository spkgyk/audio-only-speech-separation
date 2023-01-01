import torch.nn as nn

from .gc3_basics import ProjRNN, TAC

# dual-path RNN
class DPRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_group=2, dropout=0, num_layers=1, bidirectional=True, unfold=False):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_group = num_group
        self.num_spk = output_size // input_size
        self.factor = int(bidirectional) + 1
        self.unfold = unfold

        # dual-path RNN with TAC
        if self.num_group > 1:
            self.TAC = nn.ModuleList([])
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])

        if self.unfold:
            row_rnn = ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=True)
            col_rnn = ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=bidirectional)
            row_norm = nn.GroupNorm(1, input_size // num_group, eps=1e-8)
            col_norm = nn.GroupNorm(1, input_size // num_group, eps=1e-8)
            self.concat_block = nn.Sequential(
                nn.Conv2d(input_size // num_group, input_size // num_group, 1, 1, groups=input_size // num_group),
                nn.PReLU(),
            )

        for i in range(num_layers):
            if self.num_group > 1:
                self.TAC.append(TAC(input_size // num_group, hidden_size * 3 // num_group))

            self.row_rnn.append(
                row_rnn if self.unfold else ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=True)
            )
            self.col_rnn.append(
                col_rnn
                if self.unfold
                else ProjRNN(input_size // num_group, hidden_size // num_group, "LSTM", dropout, bidirectional=bidirectional)
            )
            self.row_norm.append(row_norm if self.unfold else nn.GroupNorm(1, input_size // num_group, eps=1e-8))
            self.col_norm.append(col_norm if self.unfold else nn.GroupNorm(1, input_size // num_group, eps=1e-8))

        self.output = nn.Conv2d(input_size // num_group, output_size // num_group, 1)

    def forward(self, input):
        # input shape: batch, N, dim1, dim2

        batch_size, _, dim1, dim2 = input.shape
        output = input.view(batch_size, self.num_group, -1, dim1, dim2)

        for i in range(len(self.row_rnn)):

            # GroupComm
            if self.num_group > 1:
                output = self.TAC[i](output.view(batch_size, self.num_group, -1, dim1 * dim2))  # B, G, N/G, dim1*dim2
            output = output.view(batch_size * self.num_group, -1, dim1, dim2)  # B*G, N/G, dim1, dim2

            # intra-block
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * self.num_group * dim2, dim1, -1)  # B*G*dim2, dim1, N/G
            row_output = self.row_rnn[i](row_input)  # B*G*dim2, dim1, N/G
            row_output = row_output.view(batch_size * self.num_group, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            # ^    B*G, N/G, dim1, dim2
            row_output = self.row_norm[i](row_output.view(batch_size * self.num_group, -1, dim1, dim2)).view(output.shape)
            # ^    B*G, N/G, dim1, dim2
            output = output + row_output  # B*G, N/G, dim1, dim2

            # inter-block
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * self.num_group * dim1, dim2, -1)  # B*G*dim1, dim2, N/G
            col_output = self.col_rnn[i](col_input)  # B*G*dim1, dim2, N/G
            col_output = col_output.view(batch_size * self.num_group, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            # ^    B*G, N/G, dim1, dim2
            col_output = self.col_norm[i](col_output.view(batch_size * self.num_group, -1, dim1, dim2)).view(output.shape)
            # ^    B*G, N/G, dim1, dim2
            output = self.concat_block(output + col_output) if self.unfold else output + col_output  # B*G, N/G, dim1, dim2

        output = output.view(batch_size * self.num_group, -1, dim1, dim2)
        output = self.output(output).view(batch_size, self.num_group, self.num_spk, -1, dim1, dim2)
        output = output.transpose(1, 2).contiguous()

        return output
