"""!
Adopted from https://github.com/ujscjj/DPTNet
Modified by Yi Luo {yl3364@columbia.edu}
"""
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.rnn import LSTM
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention

from .gc3_basics import TAC


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = LSTM(d_model, d_model * 2, 1, bidirectional=True)
        self.linear2 = Linear(d_model * 2 * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SingleTransformer(nn.Module):
    def __init__(self, input_size):
        super(SingleTransformer, self).__init__()

        self.transformer = TransformerEncoderLayer(d_model=input_size, nhead=4, dropout=0)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        output = self.transformer(output.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        return output


# dual-path Transformer
class DPTNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_group=16, unfold=False):
        super(DPTNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_group = num_group
        self.num_spk = output_size // input_size
        self.unfold = unfold

        # dual-path Transformer
        if self.num_group > 1:
            self.TAC = nn.ModuleList([])
        self.row_xfmr = nn.ModuleList([])
        self.col_xfmr = nn.ModuleList([])

        if self.unfold:
            row_xfmr = SingleTransformer(input_size // num_group)
            col_xfmr = SingleTransformer(input_size // num_group)
            self.concat_block = nn.Sequential(
                nn.Conv2d(input_size // num_group, input_size // num_group, 1, 1, groups=input_size // num_group),
                nn.PReLU(),
            )

        for i in range(num_layers):
            if self.num_group > 1:
                self.TAC.append(TAC(input_size // num_group, hidden_size * 3 // num_group))

            self.row_xfmr.append(row_xfmr if self.unfold else SingleTransformer(input_size // num_group))
            self.col_xfmr.append(col_xfmr if self.unfold else SingleTransformer(input_size // num_group))

        self.output = nn.Conv2d(input_size // num_group, output_size // num_group, 1)

    def forward(self, input):
        # input shape: batch, N, dim1, dim2

        batch_size, _, dim1, dim2 = input.shape
        output = input.view(batch_size, self.num_group, -1, dim1, dim2)

        for i in range(len(self.row_xfmr)):

            # GroupComm
            if self.num_group > 1:
                output = self.TAC[i](output.view(batch_size, self.num_group, -1, dim1 * dim2))  # B, G, N/G, dim1*dim2
            output = output.view(batch_size * self.num_group, -1, dim1, dim2)  # B*G, N/G, dim1, dim2

            # intra-block
            row_input = output.permute(0, 3, 2, 1).contiguous().view(batch_size * self.num_group * dim2, dim1, -1)  # B*G*dim2, dim1, N/G
            row_output = self.row_xfmr[i](row_input)  # B*G*dim2, dim1, H
            row_output = row_output.view(batch_size * self.num_group, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B*G, N, dim1, dim2
            output = output + row_output

            # inter-block
            col_input = output.permute(0, 2, 3, 1).contiguous().view(batch_size * self.num_group * dim1, dim2, -1)  # B*G*dim1, dim2, N
            col_output = self.col_xfmr[i](col_input)  # B*G*dim1, dim2, H
            col_output = col_output.view(batch_size * self.num_group, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B*G, N, dim1, dim2
            output = self.concat_block(output + col_output) if self.unfold else output + col_output

        output = output.view(batch_size * self.num_group, -1, dim1, dim2)
        output = self.output(output).view(batch_size, self.num_group, self.num_spk, -1, dim1, dim2)
        output = output.transpose(1, 2).contiguous()

        return output
