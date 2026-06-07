import torch
from torch import nn
import torch.nn.functional as F
import ptwt
import pywt
import math

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MultiHeadAttention
from recbole.model.layers import FeedForward

class CausalConv1d(nn.Module):
    """
    Lightweight causal residual block for local detail extraction.

    Key design:
    1. Left-only padding prevents future leakage.
    2. Depthwise temporal convolution captures local patterns.
    3. Pointwise mixing and residual connection stabilize training.
    """
    def __init__(self, channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=0, dilation=dilation, groups=channels, **kwargs
        )
        
        self.ln = nn.LayerNorm(channels, eps=1e-8)
        self.act = nn.GELU()

        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, L]
        res = x
        
        x_pad = F.pad(x, (self.padding, 0))
        
        out = self.conv(x_pad)        
        out = self.ln(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)        
        out = self.pointwise(out)
        
        return out + res

class LowFreqSelfAttention(nn.Module):
    """
    Low-frequency self-attention in wavelet space.

    The main residual path keeps coefficient scale unchanged.
    An internal Pre-LayerNorm is only used for QKV projection.
    """
    def __init__(self, n_heads, hidden_size, attn_dropout_prob, hidden_dropout_prob):
        super(LowFreqSelfAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        
        # Internal LayerNorm for QKV only.
        self.inner_ln = nn.LayerNorm(hidden_size)
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # Zero init makes the initial attention branch output near zero.
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: [B, L_low, H] low-frequency coefficients
            attention_mask: [B, 1, L_low, L_low] or [B, 1, 1, L_low]
        Returns:
            [B, L_low, H] residual attention output
        """
        # Normalize only before QKV projection.
        x_norm = self.inner_ln(input_tensor)
        
        mixed_query_layer = self.query(x_norm)
        mixed_key_layer = self.key(x_norm)
        mixed_value_layer = self.value(x_norm)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)
        
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        scale_factor = math.sqrt(self.dense.out_features)
        
        return input_tensor + (hidden_states / scale_factor)


    def build_mask(self, valid_mask, L_low, device):
        """
        Build the low-frequency attention mask in wavelet space.

        1. Downsample the valid mask to the low-frequency length.
        2. Optionally apply a causal mask.
        
        Args:
            valid_mask: [B, L] valid positions in time domain
            L_low: low-frequency sequence length
            device: target device
        Returns:
            attn_mask: [B, 1, L_low, L_low] attention mask
        """
        # Downsample the valid mask to the wavelet length.
        valid_float = valid_mask.float().unsqueeze(1)  # [B, 1, L]
        valid_low = F.adaptive_max_pool1d(
            valid_float, output_size=L_low
        ).squeeze(1).bool()  # [B, L_low]
        
        # Key mask: [B, 1, 1, L_low]
        key_mask = valid_low.unsqueeze(1).unsqueeze(2)
        attn_mask = torch.where(key_mask, 0.0, -10000.0)
        
        return attn_mask


class CausalDetailGating(nn.Module):
    """
    Adaptive gating for high-frequency wavelet coefficients.

    It extracts local causal context and generates input-dependent gates.
    """
    def __init__(self, channels, kernel_size=3, act_str="gelu", gate_act_str="tanh", gate_scale=1.0):
        super(CausalDetailGating, self).__init__()
        
        ACT2FN = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(), 
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        self.gate_scale = gate_scale

        self.context_conv = CausalConv1d(channels, kernel_size=kernel_size)

        self.act = ACT2FN.get(act_str.lower() , nn.GELU())
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.gate_act = ACT2FN.get(gate_act_str.lower() , nn.Tanh())

        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x):
        # x: [B, H, L]
        context = self.act(self.context_conv(x))
        # Gate range is controlled by the activation and scale.
        gate_weight = self.gate_scale * self.gate_act(self.gate_conv(context))
        return x * gate_weight

class HiWaveLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, config):
        super(HiWaveLayer, self).__init__()
        
        # Wavelet settings.
        self.wavelet_depth = config["wavelet_depth"]
        self.wavelet = pywt.Wavelet(config["wavelet"])
        self.wavelet_mode = config["wavelet_mode"]        
        self.cdg_kernel_size = config["cdg_kernel_size"]
        self.cdg_act = config["cdg_act"]
        self.cdg_gate_act = config["cdg_gate_act"]
        self.cdg_gate_scale = config["cdg_gate_scale"]

        self.high_cdgs = nn.ModuleList([
            CausalDetailGating(
                hidden_size, 
                kernel_size=self.cdg_kernel_size, 
                act_str=self.cdg_act, 
                gate_act_str=self.cdg_gate_act,
                gate_scale=self.cdg_gate_scale,
            ) for _ in range(self.wavelet_depth)
        ])

        if self.wavelet_depth == 0:
            self.attention_layer = MultiHeadAttention(
                n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
            )
        else:
            self.wavelet_attention = LowFreqSelfAttention(
                n_heads, hidden_size, attn_dropout_prob, hidden_dropout_prob
            )
            self.out_dropout = nn.Dropout(hidden_dropout_prob)
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, valid_mask):
        B, L, H = hidden_states.size()
        
        if self.wavelet_depth == 0:
            # depth=0: standard causal self-attention.
            extended_attention_mask = valid_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, L, -1))
            )
            attn_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
            
            # `MultiHeadAttention` already includes residual and LayerNorm.
            output = self.attention_layer(hidden_states, attn_mask)
            return output

        x = hidden_states.transpose(1, 2)  # [B, H, L]
        
        # Multi-level DWT decomposition.
        coeffs = ptwt.wavedec(x, self.wavelet, level=self.wavelet_depth, mode=self.wavelet_mode)
        
        # --- Low-frequency branch ---
        low_freq = coeffs[0].transpose(1, 2)
        L_low = low_freq.size(1)
        low_mask = self.wavelet_attention.build_mask(valid_mask, L_low, x.device)
        coeffs[0] = self.wavelet_attention(low_freq, low_mask).transpose(1, 2)
        
        # --- High-frequency branch ---
        for i in range(1, len(coeffs)):
            coeffs[i] = self.high_cdgs[i-1](coeffs[i])
        
        # --- IDWT reconstruction ---
        r = ptwt.waverec(coeffs, self.wavelet)
        r = r.transpose(1, 2)
        
        # Time-domain residual connection and LayerNorm.
        output = self.out_dropout(r)
        output = self.LayerNorm(output + hidden_states)
        return output

class HiWaveBlock(nn.Module):
    def __init__(self, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, config):
        super(HiWaveBlock, self).__init__()
        self.hiwave_layer = HiWaveLayer(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps, config
        )

        self.feed_forward = FeedForward(
            hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, hidden_states, valid_mask):
        layer_output = self.hiwave_layer(hidden_states, valid_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class HiWaveEncoder(nn.Module):

    def __init__(self, n_layers, n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps, config):
        super(HiWaveEncoder, self).__init__()
        # Build each block explicitly instead of using copy.deepcopy.
        self.layer = nn.ModuleList([
            HiWaveBlock(
                n_heads, hidden_size, inner_size, hidden_dropout_prob, 
                attn_dropout_prob, hidden_act, layer_norm_eps, config
            )
            for _ in range(n_layers)
        ])

    def forward(self, hidden_states, valid_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, valid_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class HiWave(SequentialRecommender):

    def __init__(self, config, dataset):
        super(HiWave, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.hiwave_encoder = HiWaveEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps,
            config=config 
        )

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        B, L = item_seq.size()
        
        # Item and position embeddings.
        pos = torch.arange(L, device=item_seq.device).expand(B, L)
        x = self.item_embedding(item_seq) + self.position_embedding(pos)
        x = self.LayerNorm(x)
        x = self.dropout(x)

        # Valid positions in the padded sequence, shape [B, L].
        valid_mask = (item_seq != 0).bool()

        trm_output = self.hiwave_encoder(x, valid_mask=valid_mask, output_all_encoded_layers=True)
        output = trm_output[-1] # [B, L, H]
        
        return self.gather_indexes(output, item_seq_len - 1)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        logits = torch.matmul(seq_output, self.item_embedding.weight.T)
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        return torch.mul(seq_output, self.item_embedding(test_item)).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return torch.matmul(seq_output, self.item_embedding.weight.T)