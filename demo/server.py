import os
import argparse

from PIL import Image
from flask import Flask, request, render_template, jsonify
import torch

from utils import base64_to_image, image_to_base64, Demonstrator


file_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(file_dir, 'template'))

demonstrator = None


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('cyclegan')
    parser.add_argument('seq2seq')
    parser.add_argument('--data-dir', default='data/crohme',
                        help='needed to create vocab.')
    parser.add_argument('--base-size', nargs=2, default=[300, 300])
    parser.add_argument('--mean', nargs='+', default=[0.5])
    parser.add_argument('--port', default=8080)
    opts = parser.parse_args()
    return opts


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    return demonstrator.demonstrate(request.data)


def main():
    global demonstrator
    opts = get_opts()
    demonstrator = Demonstrator(opts)
    app.run(host='0.0.0.0', port=opts.port)


if __name__ == "__main__":
    main()
