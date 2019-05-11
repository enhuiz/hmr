import os
import argparse
import sys
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import imageio

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchsummary import summary

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hmr.data import Vocab, MathDataset
from utils import denormalize, calculate_scores


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('cyclegan')
    parser.add_argument('seq2seq')
    parser.add_argument('data_dir')
    parser.add_argument('parts', choices=['dev', 'test'], nargs='+')
    parser.add_argument('out_dir')
    parser.add_argument('--base-size', nargs=2, default=[300, 300])
    parser.add_argument('--max-output-len', default=100)
    parser.add_argument('--mean', nargs='+', default=[0.5])
    parser.add_argument('--samples', type=int, default=20)
    opts = parser.parse_args()
    return opts


def compute_model_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class Demonstrator(object):
    def __init__(self, opts):
        self.opts = opts
        self.cyclegan = torch.load(opts.cyclegan, map_location='cpu').eval()
        self.seq2seq = torch.load(opts.seq2seq, map_location='cpu').eval()

        print('-------- #params ---------')
        print('G_XtoY', compute_model_num_params(self.cyclegan.G_XtoY))
        print('D_X', compute_model_num_params(self.cyclegan.D_X))
        print('Seq2Seq', compute_model_num_params(self.seq2seq))
        print('--------------------------')

    def torch_to_numpy(self, x):
        assert x.dim() == 3
        mean = self.opts.mean
        x = denormalize(x)
        x = x.detach().cpu().numpy().squeeze()
        return x

    def add_mask(self, image, mask, desc):
        image = self.torch_to_numpy(image)
        mask = self.torch_to_numpy(mask)
        size = image.shape

        mask = cv2.resize(mask, dsize=size)
        masked = denormalize(image + mask)
        plt.figure(figsize=(4, 3), dpi=100)
        plt.imshow(masked, cmap='gray')
        plt.title(desc, fontsize=15)
        plt.axis('off')

        with BytesIO() as buffer:
            plt.savefig(buffer, format="png")
            plt.close()
            buffer.seek(0)
            image = Image.open(buffer)
            masked = np.asarray(image)

        return masked

    def normalize(self, x):
        mean = torch.tensor(self.opts.mean)
        return x - mean

    def renormalize(self, x):
        """For the generated image, first denormalize it to range(0, 1), then normalize it by substracting the mean.
        """
        x = denormalize(x)
        x = self.normalize(x)
        return x

    def decode(self, image):
        seq2seq = self.seq2seq
        vocab = self.opts.vocab
        max_output_len = self.opts.max_output_len

        decoded = seq2seq.decode(image.unsqueeze(0),
                                 vocab.word2index('</s>'),
                                 max_output_len)

        hyp = decoded['outputs'][0].argmax(dim=-1).squeeze(1).tolist()
        hyp = vocab.decode(hyp)
        hyp = ['<s>'] + hyp + ['</s>']

        masks = list(decoded['weights'][0])
        masks += [torch.zeros_like(masks[0])]  # add empty frame for </s>
        maskeds = [self.add_mask(image, m, tok) for m, tok in zip(masks, hyp)]

        return {
            'image': self.torch_to_numpy(image),
            'maskeds': maskeds,
            'hyp': hyp,
        }

    def dump_results(self, out_dir, result):
        for k in ['real', 'fake', 'recon']:
            img = result[k]
            path = os.path.join(out_dir, '{}.png'.format(k))
            result[k] = path
            save_fig(path, img)

        for k in ['real_decoded', 'fake_decoded']:
            img = result[k]['image']
            path = os.path.join(out_dir, 'decoded', '{}.png'.format(k))
            result[k]['image'] = path
            save_fig(path, img)

            hyp = ' '.join(result[k]['hyp'])
            path = os.path.join(out_dir, 'decoded', '{}.txt'.format(k))
            result[k]['hyp_'] = path
            save_text(path, hyp)

            gif = result[k]['maskeds']
            path = os.path.join(out_dir, 'decoded', '{}.gif'.format(k))
            result[k]['maskeds'] = path
            save_gif(path, gif)

        return result

    def demonstrate(self, image, printed=False, out_dir=None):
        if os.path.exists(out_dir):
            print(out_dir, 'exists, skip.')
            return

        G_XtoY = self.cyclegan.G_XtoY
        G_YtoX = self.cyclegan.G_YtoX

        if printed:
            G_XtoY, G_YtoX = G_YtoX, G_XtoY

        real = self.normalize(image)
        fake = G_XtoY(real.unsqueeze(0))[0]
        recon = G_YtoX(fake.unsqueeze(0))[0]

        result = {
            'real': self.torch_to_numpy(real),
            'fake': self.torch_to_numpy(fake),
            'recon': self.torch_to_numpy(recon),
            'real_decoded': self.decode(real),
            'fake_decoded': self.decode(self.renormalize(fake)),
        }

        return self.dump_results(out_dir, result)


def create_transform(opts):
    return transforms.Compose([
        transforms.Resize(opts.base_size),
        transforms.ToTensor(),
        # transforms.Normalize(opts.mean, [1]), don't normalize, it will done in the demonstrator
    ])


def makedirs_for_path(path):
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)


def save_fig(path, img):
    makedirs_for_path(path)
    plt.imsave(path, img, cmap='gray')


def save_text(path, text):
    makedirs_for_path(path)
    with open(path, 'w') as f:
        f.write(text)


def save_gif(path, imgs):
    imageio.mimsave(path, imgs, fps=1)


def main():
    opts = get_opts()
    opts.vocab = Vocab(os.path.join(opts.data_dir, 'annotations', 'vocab.csv'))
    print(opts)

    demonstrator = Demonstrator(opts)

    for part in opts.parts:
        opts.part = part

        wds = MathDataset(opts.data_dir, 'written', opts.part,
                          create_transform(opts), opts.vocab)

        wdl = DataLoader(wds,
                         batch_size=1,
                         collate_fn=wds.get_collate_fn(),
                         shuffle=False)

        pds = MathDataset(opts.data_dir, 'printed', opts.part,
                          create_transform(opts), opts.vocab)

        pdl = DataLoader(pds,
                         batch_size=1,
                         collate_fn=pds.get_collate_fn(),
                         shuffle=False)

        for i, (ws, ps) in enumerate(zip(wdl, pdl)):
            assert ws['id'][0] == ps['id'][0]
            id_ = ws['id'][0]
            print('Preparing {}'.format(id_))

            written = ws['image'][0]
            out_dir = os.path.join(opts.out_dir, id_, 'written')
            result = demonstrator.demonstrate(written, False, out_dir)

            printed = ps['image'][0]
            out_dir = os.path.join(opts.out_dir, id_, 'printed')
            result = demonstrator.demonstrate(printed, True, out_dir)

            if i >= opts.samples:
                break


if __name__ == "__main__":
    main()
