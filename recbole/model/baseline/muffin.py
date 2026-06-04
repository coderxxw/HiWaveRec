"""
MUFFIN
################################################

Reference:
    "MUFFIN: Multi-Frequency Filtering Network for Sequential Recommendation"

MUFFIN uses Local Frequency Modulation (LFM) and Global Frequency Modulation (GFM)
to capture both local and global frequency patterns in user behavior sequences.
"""

import copy
import math
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish, "silu": F.silu, "sigmoid": F.sigmoid, "tanh": F.tanh}


class MUFFIN(SequentialRecommender):
    r"""
    MUFFIN: Multi-Frequency Filtering Network for Sequential Recommendation
    
    This model uses Local Frequency Modulation (LFM) and Global Frequency Modulation (GFM)
    to capture both local and global frequency patterns in user behavior sequences.
    """

    def __init__(self, config, dataset):
        super(MUFFIN, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        # MUFFIN specific parameters
        self.alpha = config["alpha"]  # auxiliary loss weight
        self.beta = config["beta"]  # load balancing loss weight
        self.kernel_size = config["kernel_size"]
        self.num_bands = config["num_bands"]
        self.freq_dropout_prob = config["freq_dropout_prob"]
        self.conv_layers = config["conv_layers"]

        # define layers
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        
        # Build config dict for encoders
        encoder_config = {
            "hidden_size": self.hidden_size,
            "inner_size": self.inner_size,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "hidden_act": self.hidden_act,
            "n_layers": self.n_layers,
            "kernel_size": self.kernel_size,
            "num_bands": self.num_bands,
            "freq_dropout_prob": self.freq_dropout_prob,
            "conv_layers": self.conv_layers,
            "MAX_ITEM_LIST_LENGTH": self.max_seq_length,
        }
        
        # LFM and GFM encoders
        self.lfm_encoder = LFMEncoder(encoder_config)
        self.gfm_encoder = GFMEncoder(encoder_config)
        
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # UAF: Unified Adaptive Filter
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
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
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def sequence_mask(self, item_seq):
        """Generate sequence mask for padding tokens"""
        mask = (item_seq != 0).float()
        return mask.unsqueeze(-1)

    def forward(self, item_seq, item_seq_len):
        seq_mask = self.sequence_mask(item_seq)
        
        # Item embeddings with masking
        item_emb = self.item_embedding(item_seq)
        item_emb = item_emb * seq_mask
        item_emb = self.LayerNorm(item_emb)
        sequence_emb = self.dropout(item_emb)
        
        # Calculate gather index
        gather_index = (item_seq_len - 1).clamp(min=0)

        # UAF: Unified Adaptive Filter in frequency domain
        frequency_emb = torch.fft.rfft(sequence_emb, dim=1, norm='ortho')
        filter_weights = torch.sigmoid(self.freq_conv_encoder(frequency_emb.abs().permute(0, 2, 1)))
        
        # GFM: Global Frequency Modulation
        gfm_layer = self.gfm_encoder(sequence_emb, seq_mask, filter_weights, output_all_encoded_layers=True)
        gfm_output = gfm_layer[-1]
        gfm_output = self.gather_indexes(gfm_output, gather_index)

        # LFM: Local Frequency Modulation
        lfm_layers, total_lb_loss = self.lfm_encoder(sequence_emb, seq_mask, filter_weights, output_all_encoded_layers=True)
        lfm_output = lfm_layers[-1]
        lfm_output = self.gather_indexes(lfm_output, gather_index)
        
        # Concatenate and fuse LFM and GFM outputs
        concat_output = torch.cat((lfm_output, gfm_output), dim=-1)
        output = self.concat_layer(concat_output)

        # Residual connection with last hidden state
        last_hidden_state = self.gather_indexes(sequence_emb, gather_index)
        output = self.LayerNorm(output + last_hidden_state)
        output = self.dropout(output)
        
        return output, gfm_output, lfm_output, total_lb_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        seq_output, gfm_output, lfm_output, total_lb_loss = self.forward(item_seq, item_seq_len)
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            
            # Main loss
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            
            # Auxiliary losses
            gfm_pos_score = torch.sum(gfm_output * pos_items_emb, dim=-1)
            gfm_neg_score = torch.sum(gfm_output * neg_items_emb, dim=-1)
            gfm_loss = self.loss_fct(gfm_pos_score, gfm_neg_score)
            
            lfm_pos_score = torch.sum(lfm_output * pos_items_emb, dim=-1)
            lfm_neg_score = torch.sum(lfm_output * neg_items_emb, dim=-1)
            lfm_loss = self.loss_fct(lfm_pos_score, lfm_neg_score)
        else:  # CE loss
            test_item_emb = self.item_embedding.weight
            
            # Main loss
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            
            # Auxiliary loss for GFM branch
            gfm_logits = torch.matmul(gfm_output, test_item_emb.transpose(0, 1))
            gfm_loss = self.loss_fct(gfm_logits, pos_items)
            
            # Auxiliary loss for LFM branch
            lfm_logits = torch.matmul(lfm_output, test_item_emb.transpose(0, 1))
            lfm_loss = self.loss_fct(lfm_logits, pos_items)
        
        # Total loss = main loss + auxiliary losses + load balancing loss
        loss = loss + self.alpha * (gfm_loss + lfm_loss)
        loss += self.beta * total_lb_loss
        
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _, _, _ = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _, _, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores


# ==================== LFM Components ====================

class LFMGate(nn.Module):
    """Gate network for Local Frequency Modulation band selection"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_bands = config["num_bands"]
        
        self.gate = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.num_bands)
        )

    def forward(self, x):
        """
        Input x: frequency domain features (Complex Tensor)
        """
        magnitude = x.abs()
        epsilon = 1e-8
        x_safe = x + torch.complex(
            torch.tensor(epsilon, device=x.device),
            torch.tensor(0.0, device=x.device)
        )
        phase = torch.angle(x_safe)
        mag_features = torch.mean(magnitude, dim=1)
        phase_features = torch.mean(phase, dim=1)
        combined_features = torch.cat([mag_features, phase_features], dim=-1)
        
        gate_logits = self.gate(combined_features)
        probs = F.softmax(gate_logits, dim=-1)
        
        local_band_prob, prob_indices = torch.topk(probs, self.num_bands, dim=-1)
        local_band_prob_normalized = local_band_prob / local_band_prob.sum(dim=-1, keepdim=True)
        
        return local_band_prob_normalized, prob_indices


class LFMFilterLayer(nn.Module):
    """Local Frequency Modulation filter layer"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_bands = config["num_bands"]
        self.kernel_size = config["kernel_size"]
        max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.hidden_size, max_seq_length // 2 + 1, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(config["freq_dropout_prob"])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.lfm_gate = LFMGate(config)
        
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )

    def compute_balance_loss(self, local_band_indices, local_band_prob):
        batch_size = local_band_indices.size(0)
        mask = F.one_hot(local_band_indices, num_classes=self.num_bands).float()
        weighted_mask = mask * local_band_prob.unsqueeze(-1)
        band_usage = weighted_mask.sum(dim=[0, 1])
        band_usage = band_usage / batch_size
        ideal_usage = torch.ones_like(band_usage) * (1 / self.num_bands)
        usage_penalty = (band_usage - ideal_usage) ** 2
        balance_loss = usage_penalty.mean()
        return balance_loss

    def forward(self, input_tensor, seq_mask, filter_weights):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        
        local_band_prob, prob_indices = self.lfm_gate(x)
        balance_loss = self.compute_balance_loss(prob_indices, local_band_prob)
        
        weight = torch.view_as_complex(self.complex_weight)
        filtered_weight = torch.complex(filter_weights * weight.real, filter_weights * weight.imag)
        x_ = x * filtered_weight.permute(0, 2, 1)

        frequency_bands = torch.empty(
            (batch, self.num_bands, max_len, hidden),
            device=input_tensor.device,
            dtype=input_tensor.dtype
        )
        
        for band in range(self.num_bands):
            frequency_output = torch.zeros_like(x_)
            band_start = band * (max_len // 2 + 1) // self.num_bands
            band_end = (band + 1) * (max_len // 2 + 1) // self.num_bands
            frequency_output[:, band_start:band_end] = x_[:, band_start:band_end]
            sequence_emb_fft = torch.fft.irfft(frequency_output, n=max_len, dim=1, norm='ortho')
            
            band_output = self.out_dropout(sequence_emb_fft)
            frequency_bands[:, band] = self.LayerNorm(band_output + input_tensor)

        selected = torch.gather(
            frequency_bands, dim=1,
            index=prob_indices.view(batch, self.num_bands, 1, 1).expand(-1, -1, max_len, hidden)
        )
        weighted_bands = local_band_prob.view(batch, self.num_bands, 1, 1) * selected
        lfm_output = weighted_bands.sum(dim=1)
        
        return lfm_output, balance_loss


class Intermediate(nn.Module):
    """Feed-forward intermediate layer"""
    
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["inner_size"])
        self.intermediate_act_fn = ACT2FN[config["hidden_act"]]
        self.dense_2 = nn.Linear(config["inner_size"], config["hidden_size"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LFMLayer(nn.Module):
    """One LFM layer"""
    
    def __init__(self, config):
        super().__init__()
        self.filter_layer = LFMFilterLayer(config)
        self.intermediate = Intermediate(config)

    def forward(self, hidden_states, seq_mask, filter_weights):
        lfm_output, balance_loss = self.filter_layer(hidden_states, seq_mask, filter_weights)
        output = self.intermediate(lfm_output)
        return output, balance_loss


class LFMEncoder(nn.Module):
    """LFM Encoder with multiple layers"""
    
    def __init__(self, config):
        super().__init__()
        layer = LFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config["n_layers"])])

    def forward(self, hidden_states, seq_mask, filter_weights, output_all_encoded_layers=True):
        all_encoder_layers = []
        total_balance_loss = 0
        
        for layer_module in self.layer:
            hidden_states, balance_loss = layer_module(hidden_states, seq_mask, filter_weights)
            total_balance_loss += balance_loss
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers, total_balance_loss


# ==================== GFM Components ====================

class GFMFilterLayer(nn.Module):
    """Global Frequency Modulation filter layer"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.kernel_size = config["kernel_size"]
        max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.hidden_size, max_seq_length // 2 + 1, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(config["freq_dropout_prob"])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect'
            ),
            nn.BatchNorm1d(self.hidden_size),
        )

    def forward(self, input_tensor, seq_mask, filter_weights):
        batch, max_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        
        filtered_weight = torch.complex(filter_weights * weight.real, filter_weights * weight.imag)
        x_ = x * filtered_weight.permute(0, 2, 1)
        
        whole_emb = torch.fft.irfft(x_, n=max_len, dim=1, norm='ortho')
        whole_emb = self.out_dropout(whole_emb)
        whole_emb = self.LayerNorm(whole_emb + input_tensor)
        return whole_emb


class GFMLayer(nn.Module):
    """One GFM layer"""
    
    def __init__(self, config):
        super().__init__()
        self.filter_layer = GFMFilterLayer(config)
        self.intermediate = Intermediate(config)

    def forward(self, hidden_states, seq_mask, filter_weights):
        gfm_output = self.filter_layer(hidden_states, seq_mask, filter_weights)
        output = self.intermediate(gfm_output)
        return output


class GFMEncoder(nn.Module):
    """GFM Encoder with multiple layers"""
    
    def __init__(self, config):
        super().__init__()
        layer = GFMLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config["n_layers"])])

    def forward(self, hidden_states, seq_mask, filter_weights, output_all_encoded_layers=True):
        all_encoder_layers = []
        
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, seq_mask, filter_weights)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return all_encoder_layers
