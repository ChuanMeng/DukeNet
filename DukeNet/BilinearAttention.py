import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearAttention(nn.Module):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.linear_key = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size=hidden_size

    def score(self, query, key, softmax_dim=-1, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        '''

        # [batch_size, *, query_seq_len, key_seq_len]
        attn=self.matching(query, key, mask)

        # [batch_size, *, query_seq_len, key_seq_len]
        norm_attn = F.softmax(attn, dim=softmax_dim)

        if mask is not None:
            norm_attn = norm_attn.masked_fill(~mask, 0)

        # attn [batch_size, *, query_seq_len, key_seq_len]
        # norm_attn [batch_size, *, query_seq_len, key_seq_len]
        return attn, norm_attn


    def matching(self, query, key, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]
        '''
        wq = self.linear_query(query) # [batch_size, *, query_seq_len, hidden_size]
        wq = wq.unsqueeze(-2) # [batch_size, *, query_seq_len, 1, hidden_size]

        uh = self.linear_key(key) # [batch_size, *, key_seq_len, hidden_size]
        uh = uh.unsqueeze(-3) # [batch_size, *, 1, key_seq_len, hidden_size]

        wuc = wq + uh  # [batch_size, *, query_seq_len, key_seq_len, hidden_size]

        wquh = torch.tanh(wuc) # [batch_size, *, query_seq_len, key_seq_len, hidden_size]

        attn = self.v(wquh).squeeze(-1) # [batch_size, *, query_seq_len, key_seq_len]

        if mask is not None:
            # masked_fill_(mask, value)。
            # The shape of mask must be broadcastable with the shape of the underlying tensor.
            attn = attn.masked_fill(~mask, -float('inf'))

        # [batch_size, *, query_seq_len, key_seq_len]
        return attn

    def forward(self, query, key, value, mask=None):
        '''
        :param query: [batch_size, *, query_seq_len, query_size]
        :param key: [batch_size, *, key_seq_len, key_size]
        :param value: [batch_size, *, value_seq_len=key_seq_len, value_size]
        :param mask: [batch_size, *, query_seq_len, key_seq_len]

        :return: [batch_size, *, query_seq_len, value_size]
        '''

        # attn [batch_size, *, query_seq_len, key_seq_len]
        # norm_attn [batch_size, *, query_seq_len, key_seq_len]
        attn, norm_attn = self.score(query, key, mask=mask)
        # torch.bmm: Performs a batch matrix-matrix product of matrices stored in input and mat2.input and mat2 must be 3-D tensors each containing the same number of matrices.
        # norm_attn.view-> [batch_size* *, query_seq_len, key_seq_len]   value.view-> [batch_size* *, key_seq_len, value_size]
        # h-> [batch_size* *, query_seq_len, value_size]
        h = torch.bmm(norm_attn.view(-1, norm_attn.size(-2), norm_attn.size(-1)), value.view(-1, value.size(-2), value.size(-1)))

        # attn [batch_size, *, query_seq_len, key_seq_len]
        # norm_attn [batch_size, *, query_seq_len, key_seq_len]
        return h.view(list(value.size())[:-2]+[norm_attn.size(-2), -1]), attn, norm_attn