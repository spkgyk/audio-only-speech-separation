import torch
import torch.nn as nn

from .utils import GC_RNN, DP_Wrapper, SudoRMRF_Wrapper, TCN_Wrapper, BaseModel, split_feature, merge_feature


class TasNet(BaseModel):
    def __init__(
        self,
        enc_dim=64,
        hidden_dim=128,
        win=2,
        layer=6,
        num_spk=2,
        module="DPRNN",
        context_size=24,
        group_size=16,
        block_size=100,
        sample_rate=16000,
    ):
        super(TasNet, self).__init__(sample_rate=sample_rate)

        assert module in [
            "DPRNN",
            "GC_DPRNN",
            "DPTNet",
            "GC_DPTNet",
            "GC_TCN",
            "TCN",
            "GC_SudoRMRF",
            "SudoRMRF",
            "Unfolded_DPRNN",
            "Unfolded_DPTNet",
        ]

        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        self.group_size = group_size
        self.win = win  # int(sample_rate * win / 1000)
        self.stride = self.win // 2
        self.model_name = module
        self.use_gc3 = "GC_" in self.model_name
        if not self.use_gc3:
            self.group_size = 1

        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)
        self.norm = nn.GroupNorm(1, self.enc_dim)

        # context encoder/decoder
        if self.use_gc3:
            self.context_enc = GC_RNN(self.enc_dim, self.hidden_dim, num_group=self.group_size, num_layers=2, bidirectional=True)
            self.context_dec = GC_RNN(self.enc_dim, self.hidden_dim, num_group=self.group_size, num_layers=2, bidirectional=True)

        # sequence modeling
        if self.model_name in ["DPRNN", "GC_DPRNN", "DPTNet", "GC_DPTNet", "Unfolded_DPRNN", "Unfolded_DPTNet"]:
            self.seq_model = DP_Wrapper(
                self.enc_dim,
                self.hidden_dim,
                self.enc_dim,
                num_spk=1,
                num_group=self.group_size,
                layer=layer,
                block_size=block_size,
                module=self.model_name,
            )
        elif self.model_name in ["TCN", "GC_TCN"]:
            self.seq_model = TCN_Wrapper(
                self.enc_dim,
                self.enc_dim,
                self.enc_dim * 4,
                layer=layer,
                stack=2,
                module=self.model_name,
                kernel=3,
                num_group=self.group_size,
                BN_dim=self.hidden_dim,
            )
        elif self.model_name in ["GC_SudoRMRF", "SudoRMRF"]:
            self.seq_model = SudoRMRF_Wrapper(
                out_channels=self.enc_dim,
                in_channels=self.hidden_dim * 2,
                upsampling_depth=5,
                layer=layer,
                module=self.model_name,
                num_group=self.group_size,
            )

        # mask estimation layer
        self.mask = nn.Sequential(
            nn.Conv1d(self.enc_dim // self.group_size, self.enc_dim * self.num_spk // self.group_size, 1), nn.ReLU(inplace=True)
        )

        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_input(self, input):
        """
        Zero-padding input according to window/stride size.
        """
        was_one_d = False
        if input.ndim == 1:
            was_one_d = True
            input = input.unsqueeze(0)
        if input.ndim == 2:
            input = input
        if input.ndim == 3:
            input = input.squeeze(1)

        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, self.stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest, was_one_d

    def forward(self, input):

        # padding
        output, rest, was_one_d = self.pad_input(input)
        batch_size = output.size(0)

        # waveform encoder
        enc_output = self.encoder(output.unsqueeze(1))  # B, N, T
        seq_len = enc_output.shape[-1]
        enc_feature = self.norm(enc_output)

        # context encoding
        if self.use_gc3:
            squeeze_block, squeeze_rest = split_feature(enc_feature, self.context_size)  # B, N, context, L
            squeeze_frame = squeeze_block.shape[-1]
            squeeze_input = (
                squeeze_block.permute(0, 3, 1, 2).contiguous().view(batch_size * squeeze_frame, self.enc_dim, self.context_size)
            )  # B*L, N, context
            squeeze_output = self.context_enc(squeeze_input)  # B*L, N, context
            squeeze_mean = squeeze_output.mean(2).view(batch_size, squeeze_frame, self.enc_dim).transpose(1, 2).contiguous()  # B, N, L
        else:
            squeeze_mean = enc_feature
            squeeze_frame = enc_feature.shape[-1]

        # sequence modeling
        feature_output = self.seq_model(squeeze_mean).view(batch_size, -1, squeeze_frame)  # B, N, L if using context encoding, else B, N, T

        # context decoding
        if self.use_gc3:
            feature_output = feature_output.unsqueeze(2) + squeeze_block  # B, N, context, L
            feature_output = (
                feature_output.permute(0, 3, 1, 2).contiguous().view(batch_size * squeeze_frame, self.enc_dim, self.context_size)
            )  # B*L, N, context
            unsqueeze_output = self.context_dec(feature_output).view(batch_size, squeeze_frame, self.enc_dim, -1)  # B, L, N, context
            unsqueeze_output = unsqueeze_output.permute(0, 2, 3, 1).contiguous()  # B, N, context, L
            unsqueeze_output = merge_feature(unsqueeze_output, squeeze_rest)  # B, N, T
        else:
            unsqueeze_output = feature_output

        # mask estimation
        unsqueeze_output = unsqueeze_output.view(batch_size * self.group_size, -1, unsqueeze_output.shape[-1])
        mask = self.mask(unsqueeze_output).view(
            batch_size, self.group_size, self.num_spk, self.enc_dim // self.group_size, -1
        )  # B, G, C, N/G, T
        mask = mask.transpose(1, 2).contiguous().view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, T
        output = mask * enc_output.unsqueeze(1)  # B, C, N, T

        # waveform decoder
        output = self.decoder(output.view(batch_size * self.num_spk, self.enc_dim, seq_len))  # B*C, 1, L
        output = output[:, :, self.stride : -(rest + self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)

        if was_one_d:
            output = output.squeeze(0)

        return output

    def get_model_args(self):
        model_args = {"n_src": 2}
        return model_args
