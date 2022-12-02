###
# Author: Kai Li
# Date: 2022-04-19 17:38:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-05-26 18:07:13
###
###
# Author: Kai Li
# Date: 2022-02-12 15:21:45
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-03-20 09:29:39
###
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from . import normalizations
from .base_model import BaseModel


class Encoder(nn.Module):
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        view_as = (-1,) + x.shape[-2:]
        x = super().forward(x.reshape(view_as)).view(x.shape[:-2] + (-1,))
        return x


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


class Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
    """

    def __init__(self, intra_mdl, inter_mdl, out_channels, norm_type):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl

        # Norm
        self.intra_norm = normalizations.get("gLN")(out_channels)
        self.inter_norm = normalizations.get(norm_type)(out_channels)

    def forward(self, x):
        """Returns the output tensor.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].
        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]
        intra = self.intra_mdl(intra)
        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        intra = self.intra_norm(intra)
        # [B, N, K, S]
        intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter = self.inter_mdl(inter)
        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.
    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model, inter_model, out_channels, norm,
                    )
                )
            )

        self.conv2d = nn.Conv2d(out_channels, out_channels * num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """Returns the output tensor.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].
        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.
        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits
        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].
        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.
        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.
        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class Sepformer(BaseModel):
    """
    Arguments
    ---------
    encoder_kernel_size: int,
        The kernel size used in the encoder
    encoder_in_nchannels: int,
        The number of channels of the input audio
    encoder_out_nchannels: int,
        The number of filters used in the encoder.
        Also, number of channels that would be inputted to the intra and inter blocks.
    masknet_chunksize: int,
        The chunk length that is to be processed by the intra blocks
    masknet_numlayers: int,
        The number of layers of combination of inter and intra blocks
    masknet_norm: str,
        The normalization type to be used in the masknet
        Should be one of 'ln' -- layernorm, 'gln' -- globallayernorm
                         'cln' -- cumulative layernorm, 'bn' -- batchnorm
                         -- see the select_norm function above for more details
    masknet_useextralinearlayer: bool,
        Whether or not to use a linear layer at the output of intra and inter blocks
    masknet_extraskipconnection: bool,
        This introduces extra skip connections around the intra block
    masknet_numspks: int,
        This determines the number of speakers to estimate
    intra_numlayers: int,
        This determines the number of layers in the intra block
    inter_numlayers: int,
        This determines the number of layers in the inter block
    intra_nhead: int,
        This determines the number of parallel attention heads in the intra block
    inter_nhead: int,
        This determines the number of parallel attention heads in the inter block
    intra_dffn: int,
        The number of dimensions in the positional feedforward model in the inter block
    inter_dffn: int,
        The number of dimensions in the positional feedforward model in the intra block
    intra_use_positional: bool,
        Whether or not to use positional encodings in the intra block
    inter_use_positional: bool,
        Whether or not to use positional encodings in the inter block
    intra_norm_before: bool
        Whether or not we use normalization before the transformations in the intra block
    inter_norm_before: bool
        Whether or not we use normalization before the transformations in the inter block
    Example
    -----
    >>> model = SepformerWrapper()
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        encoder_kernel_size=32,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_norm="LN",
        masknet_numspks=2,
        intra_numlayers=8,
        inter_numlayers=8,
        intra_nhead=8,
        inter_nhead=8,
        intra_dffn=1024,
        inter_dffn=1024,
        intra_use_positional=True,
        inter_use_positional=True,
        intra_norm_before=True,
        inter_norm_before=True,
        intra_causal=True,
        inter_causal=True,
        sample_rate=16000,
    ):

        super(Sepformer, self).__init__(sample_rate=sample_rate)
        self.encoder = Encoder(
            kernel_size=encoder_kernel_size,
            out_channels=encoder_out_nchannels,
            in_channels=encoder_in_nchannels,
        )
        intra_model = TransformerBlock(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
            is_causal=intra_causal,
        )

        inter_model = TransformerBlock(
            num_layers=inter_numlayers,
            d_model=encoder_out_nchannels,
            nhead=inter_nhead,
            d_ffn=inter_dffn,
            use_positional_encoding=inter_use_positional,
            norm_before=inter_norm_before,
            is_causal=inter_causal,
        )

        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
        )
        self.decoder = Decoder(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_in_nchannels,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            bias=False,
        )
        self.num_spks = masknet_numspks

        # reinitialize the parameters
        for module in [self.encoder, self.masknet, self.decoder]:
            self.reset_layer_recursively(module)

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the network"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def forward(self, mix):
        was_one_d = False
        if mix.ndim == 1:
            was_one_d = True
            mix = mix.unsqueeze(0).unsqueeze(1)
        if mix.ndim == 2:
            mix = mix.unsqueeze(1)

        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)
        # import pdb; pdb.set_trace()
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        # [spks, B, N, L]
        spks, B, N, L = sep_h.shape
        # [B, spks, T]
        est_source = self.decoder(sep_h.reshape(spks * B, N, L)).reshape(B, spks, -1)

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(2)
        T_est = est_source.size(2)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, T_origin - T_est))
        else:
            est_source = est_source[:, :, :T_origin]

        if was_one_d:
            return est_source.squeeze(0)
        return est_source

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
