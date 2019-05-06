import base64
import io
import os
import sys

import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from flask import jsonify

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import vocab
import matplotlib.pyplot as plt


def base64_to_image(data):
    return Image.open(io.BytesIO(base64.b64decode(data.split(b',')[1])))


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='png')
    data = b'data:image/png;base64,' + base64.b64encode(buffer.getvalue())
    return data


def tensor_to_base64(t):
    return image_to_base64(transforms.ToPILImage()(t))


def denormalize(x, opts):
    return x + torch.tensor(opts.mean).to(x.device)


def add_mask(image, weight):
    mask = F.interpolate(weight.unsqueeze(0),
                         image.shape[1:],
                         mode='bilinear', align_corners=True).squeeze()
    masked = image * (mask * 0.8 + 0.2)
    return masked


class Demonstrator(object):
    def __init__(self, opts):
        self.opts = opts

        vocab.load(opts.data_dir)
        self.cyclegan = torch.load(opts.cyclegan, map_location='cpu')
        self.seq2seq = torch.load(opts.seq2seq, map_location='cpu')

        self.transform = transforms.Compose([
            transforms.Resize(opts.base_size),
            transforms.ToTensor(),
            transforms.Normalize(opts.mean, [1]),
        ])

    def demonstrate(self, image_data):
        image = base64_to_image(image_data).convert('L')
        plt.imsave('tmp.png', image, cmap='gray')
        image = self.transform(image)
        images = image.unsqueeze(0)   # batch size 1
        fakes = self.cyclegan.G_XtoY(images)

        out = self.seq2seq.decode(fakes)
        hyp = vocab.decode(out['outputs'][0].argmax(
            dim=-1).squeeze(1).tolist())
        hyp = ['<s>'] + hyp + ['</s>']

        weights = []
        fake = denormalize(fakes[0], self.opts)
        for weight in out['weights'][0]:
            masked = add_mask(fake, weight)
            weights.append(tensor_to_base64(masked))

        return jsonify({
            'fake': tensor_to_base64(fake),
            'hyp': hyp,
            'weights': weights,
        })
