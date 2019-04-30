import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr import vocab


class MultiHeadAttn(nn.Module):
    def __init__(self, heads, d_q, d_v, d_k, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_q, d_model)
        self.v_linear = nn.Linear(d_v, d_model)
        self.k_linear = nn.Linear(d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, d_k):
        score = (q @ k.transpose(-2, -1) / d_k**0.5)
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        output = score @ v
        return output, score

    def forward(self, q, k, v):
        bs = q.size(1)
        k = self.k_linear(k).view(-1, bs, self.h, self.d_k)
        q = self.q_linear(q).view(-1, bs, self.h, self.d_k)
        v = self.v_linear(v).view(-1, bs, self.h, self.d_k)

        # swap seq and head
        k = k.transpose(0, 2)
        q = q.transpose(0, 2)
        v = v.transpose(0, 2)

        output, score = self.attention(q, k, v, self.d_k)

        output = output.transpose(0, 2).reshape(-1, bs, self.d_model)
        score = score.transpose(0, 2).contiguous()

        return output, score


class MultiHeadAttnGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(MultiHeadAttnGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.mha = MultiHeadAttn(heads, 2 * hidden_dim,
                                 input_dim, input_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def assert_sorted(l, descending=False):
        l = list(l)
        if descending:
            l = l[::-1]
        assert all(l[i] <= l[i + 1] for i in range(len(l) - 1))

    def forward(self, memory, output_len):
        """
         Args:
             memory: encoder_outputs, (memory_len, bs, input_dim), sorted by output_len
         """
        self.assert_sorted(output_len, descending=True)

        device = memory.device
        memory_len, bs = memory.shape[:2]

        output = torch.zeros(1, bs, self.output_dim).to(device)
        hidden = torch.zeros(1, bs, self.hidden_dim).to(device)

        max_output_len = max(output_len)

        outputs = torch.zeros(max_output_len, bs,
                              self.output_dim).to(device)
        hiddens = torch.zeros(max_output_len, bs,
                              self.hidden_dim).to(device)
        scores = torch.zeros(max_output_len, bs,
                             memory_len).to(device)

        for i in range(max_output_len):
            num_active = sum(i < output_len)
            output = output[:, :num_active]
            hidden = hidden[:, :num_active]
            memory = memory[:, :num_active]

            embedded = self.embedding(output.argmax(dim=2))

            query = torch.cat([embedded, hidden], dim=2)
            context, score = self.mha(query, memory, memory)
            output, hidden = self.gru(context, hidden)
            output = self.fc(output)

            outputs[i:i + 1, :num_active, :] = output
            hiddens[i:i + 1, :num_active, :] = hidden
            scores[i:i + 1, :num_active, :] = score.mean(dim=2)

        return outputs, hiddens, scores

    def decode(self, memory, max_output_len=50):
        """
        Args:
            memory, (memory_len, 1, input_dim)
        """
        device = memory.device
        memory_len = memory.shape[0]

        output = torch.zeros(1, 1, self.output_dim).to(device)
        hidden = torch.zeros(1, 1, self.hidden_dim).to(device)

        outputs = torch.zeros(max_output_len, 1,
                              self.output_dim).to(device)
        hiddens = torch.zeros(max_output_len, 1,
                              self.hidden_dim).to(device)
        scores = torch.zeros(max_output_len, 1,
                             memory_len).to(device)

        for i in range(max_output_len):
            embedded = self.embedding(output.argmax(dim=2))

            query = torch.cat([embedded, hidden], dim=2)
            context, score = self.mha(query, memory, memory)
            output, hidden = self.gru(context, hidden)
            output = self.fc(output)

            outputs[i:i + 1] = output
            hiddens[i:i + 1] = hidden
            scores[i:i + 1] = score.mean(dim=2)

            cur_word = output.argmax(dim=2).squeeze()
            if cur_word == vocab.word2index('</s>'):
                outputs = outputs[:i + 1]
                hiddens = hiddens[:i + 1]
                scores = scores[:i + 1]
                break

        return outputs, hiddens, scores