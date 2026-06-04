# -*- coding: utf-8 -*-
# @Time    : 2024
# @Author  : RecBole Baseline Implementation

"""
FMLPRec
################################################

Reference:
    Kun Zhou et al. "Filter-enhanced MLP is All You Need for Sequential Recommendation."
    In WWW 2022.

Reference code:
    https://github.com/Woeee/FMLP-Rec

"""

import copy
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import FeedForward
from recbole.model.loss import BPRLoss


class FMLPRec(SequentialRecommender):
    r"""
    FMLP-Rec replaces the self-attention mechanism with learnable filters in the frequency domain.
    
    The model applies FFT to transform sequences into frequency domain, multiplies with learnable
    complex weights (filters), and transforms back to time domain. This approach captures global
    dependencies with linear complexity.
    """

    def __init__(self, config, dataset):
        super(FMLPRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        # FMLP encoder with frequency-domain filtering
        self.fmlp_encoder = FMLPRecEncoder(
            n_layers=self.n_layers,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            max_seq_length=self.max_seq_length,
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

        output = self.fmlp_encoder(input_emb, output_all_encoded_layers=True)
        output = output[-1]
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


class FilterLayer(nn.Module):
    """
    Learnable Filter Layer using FFT.
    
    Applies FFT to transform sequences into frequency domain, multiplies with learnable
    complex weights, and transforms back. This replaces self-attention with O(n log n) complexity.
    """
    
    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps, max_seq_length):
        super(FilterLayer, self).__init__()
        # Learnable complex weights for frequency filtering
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_tensor):
        # input_shape: [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        
        # Apply FFT along sequence dimension
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        # Apply learnable complex filter
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        
        # Transform back to time domain
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor  # residual connection
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class FMLPRecBlock(nn.Module):
    """
    One FMLP block consists of a FilterLayer and a point-wise feed-forward layer.
    """
    
    def __init__(
        self,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        hidden_act,
        layer_norm_eps,
        max_seq_length,
    ):
        super(FMLPRecBlock, self).__init__()
        self.filter_layer = FilterLayer(
            hidden_size, hidden_dropout_prob, layer_norm_eps, max_seq_length
        )
        self.feed_forward = FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states):
        filter_output = self.filter_layer(hidden_states)
        feedforward_output = self.feed_forward(filter_output)
        return feedforward_output


class FMLPRecEncoder(nn.Module):
    r"""
    FMLP-Rec Encoder consists of multiple stacked FMLPRecBlocks.
    
    Args:
        n_layers: number of FMLP blocks
        hidden_size: hidden dimension
        inner_size: feed-forward inner dimension
        hidden_dropout_prob: dropout probability
        hidden_act: activation function
        layer_norm_eps: layer norm epsilon
        max_seq_length: maximum sequence length (for filter initialization)
    """

    def __init__(
        self,
        n_layers=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        max_seq_length=50,
    ):
        super(FMLPRecEncoder, self).__init__()
        block = FMLPRecBlock(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
            max_seq_length,
        )
        self.layer = nn.ModuleList([copy.deepcopy(block) for _ in range(n_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        """
        Args:
            hidden_states: the input of the FMLPRecEncoder
            output_all_encoded_layers: whether to output all layers' output

        Returns:
            all_encoder_layers: list of encoder outputs
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
