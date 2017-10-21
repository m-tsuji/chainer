import unittest

import numpy

import chainer
from chainer import caffe_export
import chainer.functions as F
import chainer.links as L
from chainer import testing


# @testing.parameterize([
#     {'layer': 'LinearFunction'},
#     {'layer': 'Reshape'},
#     {'layer': 'Convolution2DFunction'},
#     {'layer': 'AveragePooling2D'},
#     {'layer': 'MaxPooling2D'},
#     {'layer': 'BatchNormalization'},
#     {'layer': 'ReLU'},
#     {'layer': 'Softmax'},
#     {'layer': 'Add'},
# ])
class TestCaffeExport(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.l1 = L.Convolution2D(None, 1, 1, 1, 0)
                    self.b2 = L.BatchNormalization(1)
                    self.l3 = L.Linear(None, 1)

            def __call__(self, x):
                h = F.relu(self.l1(x))
                h = self.b2(h)
                return self.l3(h)

        self.model = Model()

    def test_caffe_export_no_save(self):
        x = numpy.ones((1, 3, 7, 7)).astype(numpy.float32)
        with chainer.using_config('train', False), \
                chainer.force_backprop_mode():
            y = self.model(x)

        caffe_export([x], [y], None, True, 'test')


testing.run_module(__name__, __file__)
