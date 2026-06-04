# -*- coding: utf-8 -*-
"""
WEARec
################################################

Reference:
    "WEARec: Wavelet-Enhanced Adaptive Recommendation."

Reference code:
    https://github.com/WEARec

"""

import copy
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward
from recbole.model.loss import BPRLoss


class WEARec(SequentialRecommender):
    r"""
    WEARec combines FFT-based global frequency filtering with Haar wavelet local
    feature extraction, replacing self-attention with a dual-path spectral approach.

    The model uses:
    1. FFT global path: Adaptive frequency-domain filtering with learnable base filter + MLP modulation
    2. Haar wavelet local path: Single-level Haar DWT with learnable detail weights
    3. Gate fusion: Learnable alpha blending of both paths
    """

    def __init__(self, config, dataset):
        super(WEARec, self).__init__(config, dataset)

        # load standard parameters
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # WEARec specific parameters
        self.alpha = config["alpha"]
        self.adaptive = config["adaptive"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        # WEARec encoder
        self.wea_encoder = WEARecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_seq_length=self.max_seq_length,
            alpha=self.alpha,
            adaptive=self.adaptive,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.wea_encoder(
            input_emb, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class WEARecLayer(nn.Module):
    """
    WEARec core layer: dual-path spectral processing.

    Path 1 (Global): rFFT → adaptive filter modulation → iFFT
    Path 2 (Local):  Haar wavelet decomposition → learnable detail weights → reconstruction
    Fusion: gate blending with alpha

    Args:
        n_heads: number of attention heads
        hidden_size: total hidden dimension
        hidden_dropout_prob: dropout probability
        layer_norm_eps: layer norm epsilon
        max_seq_length: maximum sequence length
        alpha: mixing ratio between wavelet and FFT (0=all wavelet, 1=all FFT)
        adaptive: whether to use adaptive frequency modulation
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        layer_norm_eps,
        max_seq_length,
        alpha=0.5,
        adaptive=True,
    ):
        super(WEARecLayer, self).__init__()

        if hidden_size % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")

        self.num_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.hidden_size = hidden_size
        self.seq_len = max_seq_length
        self.adaptive = adaptive
        self.alpha = alpha

        # Frequency bins for rFFT
        self.freq_bins = max_seq_length // 2 + 1

        # ---- FFT global path parameters ----
        # Base multiplicative filter: one per head and frequency bin.
        self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
        # Base additive bias
        self.base_bias = nn.Parameter(
            torch.full((self.num_heads, self.freq_bins, 1), -0.1)
        )

        if adaptive:
            # Adaptive MLP: produces 2 values per head & frequency bin (scale and bias modulation).
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.num_heads * self.freq_bins * 2),
            )

        # ---- Haar wavelet local path parameters ----
        # Learnable weight for wavelet detail coefficients
        self.complex_weight = nn.Parameter(
            torch.randn(
                1, self.num_heads, max_seq_length // 2, self.head_dim, dtype=torch.float32
            )
            * 0.02
        )

        # ---- Output ----
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def wavelet_transform(self, x_heads):
        """
        Applies a single-level Haar wavelet transform (decomposition + reconstruction)
        to capture local dependencies along the sequence dimension.

        Args:
            x_heads: Tensor of shape (B, num_heads, seq_len, head_dim)

        Returns:
            Reconstructed wavelet-based features of the same shape.
        """
        B, H, N, D = x_heads.shape

        # For simplicity, if N is odd, truncate by one
        N_even = N if (N % 2) == 0 else (N - 1)
        x_heads = x_heads[:, :, :N_even, :]

        # Split even and odd positions
        x_even = x_heads[:, :, 0::2, :]  # (B, H, N_even/2, D)
        x_odd = x_heads[:, :, 1::2, :]

        # Haar wavelet decomposition
        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)

        # Apply learnable weight to detail coefficients
        detail = detail * self.complex_weight

        # Haar wavelet reconstruction
        x_even_recon = approx + detail
        x_odd_recon = approx - detail

        # Interleave even/odd back to original shape
        out = torch.zeros(
            (B, H, N_even, D), device=x_heads.device, dtype=x_heads.dtype
        )
        out[:, :, 0::2, :] = x_even_recon
        out[:, :, 1::2, :] = x_odd_recon

        # If we truncated one position, pad it back with zeros
        if N_even < N:
            pad = torch.zeros((B, H, 1, D), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=2)

        return out

    def forward(self, input_tensor):
        # input_tensor: [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # Reshape to separate heads: (B, num_heads, seq_len, head_dim)
        x_heads = input_tensor.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # ---- (1) FFT-based global features ----
        F_fft = torch.fft.rfft(x_heads, dim=2, norm="ortho")

        # Compute adaptive modulation parameters if enabled.
        if self.adaptive:
            context = input_tensor.mean(dim=1)  # (B, hidden)
            adapt_params = self.adaptive_mlp(context)  # (B, num_heads*freq_bins*2)
            adapt_params = adapt_params.view(
                batch, self.num_heads, self.freq_bins, 2
            )
            adaptive_scale = adapt_params[..., 0:1]
            adaptive_bias = adapt_params[..., 1:2]
        else:
            adaptive_scale = torch.zeros(
                batch, self.num_heads, self.freq_bins, 1, device=input_tensor.device
            )
            adaptive_bias = torch.zeros(
                batch, self.num_heads, self.freq_bins, 1, device=input_tensor.device
            )

        # Combine base parameters with adaptive modulations
        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias

        # Apply modulations in the frequency domain
        F_fft_mod = F_fft * effective_filter + effective_bias

        # Inverse FFT
        x_fft = torch.fft.irfft(
            F_fft_mod, dim=2, n=self.seq_len, norm="ortho"
        )  # (B, num_heads, seq_len, head_dim)

        # ---- (2) Wavelet-based local features ----
        x_wavelet = self.wavelet_transform(x_heads)

        # ---- (3) Gate fusion ----
        x_combined = (1.0 - self.alpha) * x_wavelet + self.alpha * x_fft

        # Reshape: merge heads back
        x_out = x_combined.permute(0, 2, 1, 3).reshape(batch, seq_len, hidden)

        hidden_states = self.out_dropout(x_out)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class WEARecBlock(nn.Module):
    """One WEARec block consists of a WEARecLayer and a point-wise feed-forward layer."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
        max_seq_length,
        alpha,
        adaptive,
    ):
        super(WEARecBlock, self).__init__()
        self.wea_layer = WEARecLayer(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
            max_seq_length,
            alpha,
            adaptive,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states):
        layer_output = self.wea_layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class WEARecEncoder(nn.Module):
    r"""
    WEARec Encoder consists of multiple stacked WEARecBlocks.

    Args:
        n_layers: number of WEARec blocks
        n_heads: number of multi-heads
        hidden_size: hidden dimension
        inner_size: feed-forward inner dimension
        hidden_dropout_prob: dropout probability
        hidden_act: activation function
        layer_norm_eps: layer norm epsilon
        max_seq_length: maximum sequence length
        alpha: mixing ratio between wavelet and FFT
        adaptive: whether to use adaptive frequency modulation
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        max_seq_length=50,
        alpha=0.5,
        adaptive=True,
    ):
        super(WEARecEncoder, self).__init__()
        block = WEARecBlock(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            max_seq_length,
            alpha,
            adaptive,
        )
        self.layer = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(n_layers)]
        )

    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
