"""
BSARec
################################################

Reference:
    Yehjin Shin et al. "BSARec: Bilateral Self-Augmented Recommendation with Spectral-Domain Enhancement."
    In SIGIR 2024.

Reference code:
    https://github.com/yehjin-shin/BSARec

"""

import copy
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MultiHeadAttention, FeedForward
from recbole.model.loss import BPRLoss


class BSARec(SequentialRecommender):
    r"""
    BSARec combines frequency-domain filtering with self-attention for sequential recommendation.
    
    The model uses a bilateral approach that integrates:
    1. Dense Spectral Processing (DSP): Low-pass filtering in frequency domain with learnable high-frequency gain
    2. Global Spectral Processing (GSP): Standard multi-head self-attention
    
    These two paths are combined with a learnable mixing parameter alpha.
    """

    def __init__(self, config, dataset):
        super(BSARec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        # BSARec specific parameters
        self.c = config["c"]  # cutoff frequency for low-pass filter
        self.alpha = config["alpha"]  # mixing ratio between DSP and GSP

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # BSARec encoder with frequency-domain processing
        self.bsa_encoder = BSARecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            c=self.c,
            alpha=self.alpha,
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

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.bsa_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
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


class FrequencyLayer(nn.Module):
    """
    Dense Spectral Processing (DSP) layer using FFT-based frequency filtering.
    
    Original BSARec implementation:
        1. Compute low_pass via FFT → zero high freq → IFFT
        2. high_pass = input - low_pass (time domain subtraction)
        3. output = low_pass + sqrt_beta² × high_pass
        4. LayerNorm(output + input) (residual)
    """
    
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.c = c // 2 + 1  # cutoff frequency index
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # input_shape: [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        
        # FFT
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        
        # Zero out high frequencies to get low_pass (in frequency domain)
        low_pass_freq = x.clone()
        low_pass_freq[:, self.c:, :] = 0
        
        # IFFT to get low_pass in time domain
        low_pass = torch.fft.irfft(low_pass_freq, n=seq_len, dim=1, norm='ortho')
        
        # Compute high_pass in TIME DOMAIN (original BSARec method)
        high_pass = input_tensor - low_pass
        
        # Combine: low_pass + beta² × high_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BSARecLayer(nn.Module):
    """
    BSARec Layer that combines Dense Spectral Processing (DSP) and Global Spectral Processing (GSP).
    
    DSP: Frequency-domain filtering for capturing periodic patterns
    GSP: Multi-head self-attention for capturing global dependencies
    
    The outputs are combined with a mixing parameter alpha.
    """
    
    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        c,
        alpha,
    ):
        super(BSARecLayer, self).__init__()
        self.alpha = alpha
        
        # Dense Spectral Processing (frequency domain)
        self.frequency_layer = FrequencyLayer(
            hidden_size, hidden_dropout_prob, layer_norm_eps, c
        )
        
        # Global Spectral Processing (self-attention)
        self.attention_layer = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )

    def forward(self, hidden_states, attention_mask):
        # DSP path: frequency-domain filtering
        dsp_output = self.frequency_layer(hidden_states)
        
        # GSP path: multi-head self-attention
        gsp_output = self.attention_layer(hidden_states, attention_mask)
        
        # Combine both paths with alpha mixing
        output = self.alpha * dsp_output + (1 - self.alpha) * gsp_output
        
        return output


class BSARecBlock(nn.Module):
    """
    One BSARec block consists of a BSARecLayer and a point-wise feed-forward layer.
    """
    
    def __init__(
        self,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        c,
        alpha,
    ):
        super(BSARecBlock, self).__init__()
        self.bsa_layer = BSARecLayer(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            c,
            alpha,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        layer_output = self.bsa_layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class BSARecEncoder(nn.Module):
    r"""
    BSARec Encoder consists of multiple stacked BSARecBlocks.
    
    Args:
        n_layers: number of BSARec blocks
        n_heads: number of attention heads
        hidden_size: hidden dimension
        inner_size: feed-forward inner dimension
        hidden_dropout_prob: dropout probability
        attn_dropout_prob: attention dropout probability
        hidden_act: activation function
        layer_norm_eps: layer norm epsilon
        c: cutoff frequency for low-pass filter
        alpha: mixing ratio between DSP and GSP
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        c=1,
        alpha=0.5,
    ):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            c,
            alpha,
        )
        self.layer = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: the input of the BSARecEncoder
            attention_mask: the attention mask for the input
            output_all_encoded_layers: whether to output all layers' output

        Returns:
            all_encoder_layers: list of encoder outputs
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
