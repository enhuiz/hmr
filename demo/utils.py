import base64
import io
import os
import sys
from torchsummary import summary

import numpy as np
import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from flask import jsonify

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import matplotlib.pyplot as plt


def base64_to_pil(data):
    return Image.open(io.BytesIO(base64.b64decode(data.split(b',')[1])))


def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='png')
    data = b'data:image/png;base64,' + base64.b64encode(buffer.getvalue())
    data = data.decode('utf-8')
    return data


def add_mask(image, weight):
    mask = F.interpolate(weight.unsqueeze(0),
                         image.shape[1:],
                         mode='bilinear', align_corners=True).squeeze()
    masked = image * (mask * 0.8 + 0.2)
    return masked


def compute_model_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class Demonstrator(object):
    def __init__(self, opts):
        self.opts = opts

        self.cyclegan = torch.load(opts.cyclegan, map_location='cpu').eval()
        self.seq2seq = torch.load(opts.seq2seq, map_location='cpu').eval()

        print(compute_model_num_params(self.cyclegan.G_XtoY))
        print(compute_model_num_params(self.cyclegan.D_X))
        print(compute_model_num_params(self.seq2seq))

        self.transform = transforms.Compose([
            transforms.Resize(opts.base_size),
            transforms.ToTensor(),
            transforms.Normalize(opts.mean, [1]),
        ])

    def tensor_to_pil(self, img):
        img = img.detach().cpu().numpy()
        img = img.squeeze() * 255
        img = Image.fromarray(img)
        img = img.convert('L')
        return img

    def tensor_to_base64(self, img):
        return pil_to_base64(self.tensor_to_pil(img))

    def demonstrate(self, image_data):
        opts = self.opts

        image = base64_to_pil(image_data).convert('L')
        # image = Image.open(
        #     'data/crohme/features/written/dev/18_em_0.png').convert('L')
        plt.imsave('tmp.png', image, cmap='gray')

        image = self.transform(image)
        images = image.unsqueeze(0)   # batch size 1
        fake = self.cyclegan.G_XtoY(images)[0]

        # change fake to L then get it back
        fake += torch.tensor(opts.mean)
        fake = self.tensor_to_pil(fake)
        fake = self.transform(fake)

        fakes = fake.unsqueeze(0)
        with torch.no_grad():
            out = self.seq2seq.decode(fakes,
                                      opts.vocab.word2index('</s>'),
                                      opts.max_output_len)

        hyp = opts.vocab.decode(out['outputs'][0].argmax(
            dim=-1).squeeze(1).tolist())

        hyp = ['<s>'] + hyp + ['</s>']

        fake += torch.tensor(opts.mean)
        weights = []
        for weight in out['weights'][0]:
            masked = add_mask(fake, weight)
            weights.append(self.tensor_to_base64(masked))

        return jsonify({
            'fake': self.tensor_to_base64(fake),
            'hyp': hyp,
            'weights': weights,
        })
