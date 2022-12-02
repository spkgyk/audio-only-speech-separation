###
# Author: Kai Li
# Date: 2022-02-12 15:21:45
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-05-26 18:07:04
###
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from .base_model import BaseModel
from .base_model import BaseModel
from torch.autograd import Variable



class PositionalEncoding(nn.Module):
    """
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class MultiheadAttention(nn.Module):
    """The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.
    Reference: https://pytorch.org/docs/stable/nn.html
    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).
    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        nhead,
        d_model,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[torch.Tensor] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        pos_embs: torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).
        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        output = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
        )

        if return_attn_weights:
            output, attention_weights = output
            # reshape the output back to (batch, time, fea)
            output = output.permute(1, 0, 2)
            return output, attention_weights
        else:
            output = output.permute(1, 0, 2)
            return output


class PositionalwiseFeedForward(nn.Module):
    """The class implements the positional-wise feed forward module in
    âAttention Is All You Needâ.
    Arguments
    ----------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).
    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self, d_ffn, input_shape=None, input_size=None, dropout=0.0, activation=nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.ffn = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, input_size),
        )

    def forward(self, x):
        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
    ):
        super().__init__()

        self.self_att = MultiheadAttention(
            nhead=nhead, d_model=d_model, dropout=dropout, kdim=kdim, vdim=vdim,
        )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation,
        )

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output, self_attn


class TransformerEncoder(nn.Module):
    """This class implements the transformer encoder.
    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst


class TransformerBlock(nn.Module):
    """
    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        is_causal=False,
    ):
        super(TransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model)

        self.is_causal = is_causal

    def forward(self, x):
        """Returns the transformed output.
        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
        """
        att_mask = self.get_lookahead_mask(x) if self.is_causal else None
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            return self.mdl(x + pos_enc, src_mask=att_mask)[0]
        else:
            return self.mdl(x, src_mask=att_mask)[0]

    def get_lookahead_mask(self, padded_input):
        """Creates a binary mask for each sequence which maskes future frames.
        Arguments
        ---------
        padded_input: torch.Tensor
            Padded input tensor.
        Example
        -------
        >>> a = torch.LongTensor([[1,1,0], [2,3,0], [4,5,0]])
        >>> get_lookahead_mask(a)
        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0.]])
        """
        seq_len = padded_input.shape[1]
        mask = (
            torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device)) == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.detach().to(padded_input.device)


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
                    TransformerBlock(
                        num_layers=num_layers,
                        d_model=input_size,
                        nhead=8,
                        d_ffn=hidden_size,
                        use_positional_encoding=True,
                        norm_before=True,
                        is_causal=False,
                    )
                )
                self.col_rnn.append(
                    TransformerBlock(
                        num_layers=num_layers,
                        d_model=input_size,
                        nhead=8,
                        d_ffn=hidden_size,
                        use_positional_encoding=True,
                        norm_before=True,
                        is_causal=False,
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

        assert model_type in [
            "DPRNN",
            "DPRNN_TAC",
            "GCDPRNN",
        ], "model_type can only be 'DPRNN', 'DPRNN_TAC', or 'GCDPRNN'."

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

        # DPRNN model
        if model_type == "GCDPRNN":
            self.DPRNN = getattr(sys.modules[__name__], model_type)(
                self.feature_dim,
                self.hidden_dim,
                self.output_dim * self.num_spk,
                num_layers=layer,
                bidirectional=bidirectional,
                group=group,
                full_causal=full_causal,
            )
        else:
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


class Sepformer(BaseModel):
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
        super(Sepformer, self).__init__(sample_rate=sample_rate)

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
