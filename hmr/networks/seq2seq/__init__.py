import torch
import torch.nn as nn
import random

from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from hmr.data import vocab
from .encoder import NLayerD, ResNetEncoder
from .decoder import MultiHeadAttnRNN


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None, y_len=None):
        x = self.encoder(x)
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)  # (bs, c, h, w) -> (bs, c, hw)
        x = x.permute(2, 0, 1)  # (hw, bs, c)

        y_len, idx = y_len.sort(dim=0, descending=True)
        x = x[:, idx]
        y = y[:, idx]

        outputs, _, weights = self.decoder(x, y_len)
        weights = weights.reshape(-1, bs, h, w)

        outputs, _ = pack_padded_sequence(outputs, y_len)
        y, _ = pack_padded_sequence(y, y_len)
        loss = self.criterion(outputs, y)

        logp = F.log_softmax(outputs, dim=-1)

        return {
            'logp': logp,
            'loss': loss,
            'weights': weights,
        }

    def decode(self, x, max_output_len=100):
        x = self.encoder(x)
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)  # (bs, c, h, w) -> (bs, c, hw)
        x = x.permute(2, 0, 1)  # (hw, bs, c)

        outputs, _, weights = self.decoder.decode(x, max_output_len)
        weights = weights.reshape(-1, bs, h, w)

        logp = F.log_softmax(outputs, dim=-1)

        return {
            'logp': logp,
            'weights': weights,
        }


class Seq2Seq(EncoderDecoder):
    def __init__(self, opts):
        super().__init__()
        # NLayerD(1, output_nc=opts.decoder.input_dim)
        self.encoder = ResNetEncoder(opts.encoder)
        self.decoder = MultiHeadAttnRNN(opts.decoder, len(vocab))
