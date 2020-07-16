import torch
import copy
from torch.nn.modules.container import ModuleList
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required). hidden_size
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory1, memory2, tgt_mask=None, memory_mask1=None, memory_mask2=None,
                tgt_key_padding_mask=None, memory_key_padding_mask1=None, memory_key_padding_mask2=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt = self.norm1(tgt)
        # attn_output (T, N, E)
        # attn_output_weights (N, T, T)
        tgt2, tgt_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)

        # add&norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)

        tgt2, memory_weights1 = self.multihead_attn1(tgt, memory1, memory1, attn_mask=memory_mask1,
                                   key_padding_mask=memory_key_padding_mask1)
        # add&norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)

        tgt2, memory_weights2 = self.multihead_attn2(tgt, memory2, memory2, attn_mask=memory_mask2,
                                   key_padding_mask=memory_key_padding_mask2)

        # add&norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm4(tgt)


        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))

        tgt = tgt + self.dropout4(tgt2)

        # (T, N, E)   (N, T, T)  (N, T, M)
        return tgt, tgt_weights, memory_weights1, memory_weights2

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, tgt, memory1, memory2, tgt_mask = None, memory_mask1 = None, memory_mask2 = None,
    tgt_key_padding_mask = None, memory_key_padding_mask1 = None, memory_key_padding_mask2 = None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output, output_weights, memory_weights1, memory_weights2 = self.layers[i](output, memory1, memory2, tgt_mask=tgt_mask,
                                    memory_mask1=memory_mask1,
                                    memory_mask2=memory_mask2,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask1=memory_key_padding_mask1,
                                    memory_key_padding_mask2=memory_key_padding_mask2)

        if self.norm:
            output = self.norm(output)

        return output, output_weights, memory_weights1, memory_weights2