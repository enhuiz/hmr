import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torchsummary

from hmr.networks.utils import weights_init
from .encoder import NLayerD, ResNetEncoder
from .decoder import MultiHeadAttnRNN


def positional_encoding(n_position, emb_dim):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

    # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    # apply cos on 1st,3rd,5th...emb_dim
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class EncoderDecoderBase(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.pe = positional_encoding(
            10000, opts.encoder.output_dim)

    def feature_map_to_sequence(self, x, bs, h, w):
        x = x.reshape(len(x), -1, h * w)  # (bs, c, h, w) -> (bs, c, hw)
        x = x.transpose(1, 2)  # (bs, hw, c)
        x = x + self.pe[:h * w].to(x.device)
        x = x.transpose(0, 1)  # (hw, bs, c)
        return x

    def sequence_to_feature_map(self, x, bs, h, w):
        x = x.reshape(-1, bs, h, w)
        return x

    def forward(self, x, y=None, y_len=None):
        x = self.encoder(x)

        bs, _, h, w = x.shape
        x = self.feature_map_to_sequence(x, bs, h, w)

        y_len, idx = y_len.sort(dim=0, descending=True)
        x, y = x[:, idx], y[:, idx]

        outputs, _, weights = self.decoder(x, y, y_len)

        outputs = pack_padded_sequence(outputs, y_len)[0]
        y = pack_padded_sequence(y, y_len)[0]
        loss = self.criterion(outputs, y)

        weights = self.sequence_to_feature_map(weights, bs, h, w)

        return {
            'outputs': outputs,
            'weights': weights,
            'loss': loss,
        }

    def decode(self, x, eos, max_output_len):
        x = self.encoder(x)
        bs, _, h, w = x.shape

        x = self.feature_map_to_sequence(x, bs, h, w)
        outputs_list = []
        weights_list = []

        for i in range(bs):
            outputs, _, weights = self.decoder.decode(
                x[:, i:i+1], eos, max_output_len)
            weights = self.sequence_to_feature_map(weights, 1, h, w)

            outputs_list.append(outputs)
            weights_list.append(weights)

        return {
            'outputs': outputs_list,
            'weights': weights_list,
        }


class Seq2Seq(EncoderDecoderBase):
    def __init__(self, opts):
        super().__init__(opts)
        # NLayerD(1, output_nc=opts.decoder.input_dim)
        # self.encoder = ResNetEncoder(opts.encoder)
        self.encoder = NLayerD(1, output_nc=opts.decoder.input_dim)
        self.decoder = MultiHeadAttnRNN(opts.decoder, len(opts.vocab))
        self.apply(weights_init('gaussian'))
