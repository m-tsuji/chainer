import argparse
import os

import chainer
from chainer.links.model.vision import googlenet
from chainer.links.model.vision import resnet
from chainer.links.model.vision import vgg
import numpy as np


archs = {
    'googlenet': googlenet.GoogLeNet,
    'resnet50': resnet.ResNet50Layers,
    'resnet101': resnet.ResNet101Layers,
    'resnet152': resnet.ResNet152Layers,
    'vgg16': vgg.VGG16Layers,
}


def get_network_for_imagenet(arch_name):
    model = archs[arch_name]()
    input_image = np.ones((1, 3, 244, 244), dtype='f')
    with chainer.force_backprop_mode(), chainer.using_config('train', False):
        x = chainer.Variable(input_image, requires_grad=True)
        y = model(x, layers=['prob'])['prob']
    return input_image, model, y


def main():
    parser = argparse.ArgumentParser(description='Export')
    parser.add_argument(
        '--arch', '-a', type=str, required=True,
        help='Arch name. models: ' + ', '.join(archs.keys()) + '.')
    parser.add_argument(
        '--out-dir', '-o', type=str, required=True,
        help='Output directory name. '
             'chainer_model.prototxt, chainer_model.caffemodel'
             ' will be created in it')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        print('Created output directory: ' + args.out_dir)
        os.mkdir(args.out_dir)
    else:
        print('Overwriting the existing directory: ' + args.out_dir)
    if not os.path.isdir(args.out_dir):
        raise ValueError(args.out_dir + ' exists but not a directory!')

    print("load model")
    input, model, output = get_network_for_imagenet(args.arch)
    print("convert to caffe model")
    chainer.caffe_export([input], [output], args.out_dir, True)


if __name__ == '__main__':
    main()
