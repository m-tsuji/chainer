from collections import OrderedDict

import numpy
import os
import six

from chainer import function
from chainer import function_node
from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
from chainer import variable


_function_types = (function.Function, function_node.FunctionNode)


def _add_blob(layer, shape, data):
    # The following part is ridiculously slow!!
    # TODO(who): Replace with C++ extension call
    blob = layer.blobs.add()
    blob.shape.dim[:] = shape
    blob.data[:] = data.flatten()


class _RetrieveAsCaffeModel(object):

    debug = False

    def __init__(self, prototxt, caffemodel=None):
        self.caffemodel = caffemodel
        self.prototxt = prototxt

        self.seen_funcs = set()
        self.naming_map = {}  # key:string, val:dict(key: func, val: index)

    def _get_layer_name(self, layer):
        """Generate layer name like "Convolution2DFunction-10-2".

        The first number means rank of the layer (depth from the top),
        and the second number is for preventing duplication
        (different layer objects can have same rank)

        :param layer: Function object
        :return: string unique to functions

        """
        label = '{}-{}'.format(layer.label, layer.rank)
        if label not in self.naming_map:
            self.naming_map[label] = {}

        if layer not in self.naming_map[label].keys():
            self.naming_map[label][layer] = len(self.naming_map[label]) + 1
        return '{}-{}'.format(label, len(self.naming_map[label]))

    def _get_parent_name(self, parent_):
        if parent_ is None:
            # return 'input'
            return 'data'
        return self._get_layer_name(parent_)

    def _gen_layer_prototxt(self, layer_params, name='layer', depth=0,
                            indent=2):
        if type(layer_params) in (dict, OrderedDict):
            s = name + ' {\n'
            indent_s = ' ' * ((depth + 1) * indent)
            for key, val in layer_params.items():
                s += indent_s + \
                    self._gen_layer_prototxt(val, name=key, depth=depth + 1)
            s += ' ' * (depth * indent)
            s += '}\n'
            return s
        elif type(layer_params) in (int, float):
            return '{}: {}\n'.format(name, layer_params)
        elif type(layer_params) is bool:
            return '{}: {}\n'.format(name, 'true' if layer_params else 'false')
        elif type(layer_params) is str:
            return '{}: "{}"\n'.format(name, layer_params)
        elif type(layer_params) is list:
            s = ''
            indent_s = ' ' * depth * indent
            for i, t in enumerate(layer_params):
                if i != 0:
                    s += indent_s
                s += self._gen_layer_prototxt(t, name=name, depth=depth + 1)
            return s
        else:
            raise ValueError('Unsupported type: ' + str(type(val)))

    def dump_function_object(self, func, prototxt, net):
        assert isinstance(func, _function_types)
        layer_name = self._get_layer_name(func)
        parent_layer_names = [self._get_parent_name(input_.creator)
                              for input_ in func.inputs]
        params = OrderedDict()
        params['type'] = None
        params['name'] = layer_name
        params['top'] = layer_name
        params['bottom'] = parent_layer_names
        layer = None
        if net is not None:
            layer = net.layer.add()
        if func.label == 'LinearFunction':
            if len(func.inputs) == 2:
                _, W = func.inputs
                b = None
            else:
                _, W, b = func.inputs
            n_out, n_in = W.shape
            inner_product_param = {
                'num_output': n_out,
                'bias_term': b is not None,
            }
            params['type'] = 'InnerProduct'
            params['inner_product_param'] = inner_product_param
            params['bottom'] = params['bottom'][:1]

            if net is not None:
                for k, v in six.iteritems(inner_product_param):
                    setattr(layer.inner_product_param, k, v)
                _add_blob(layer, list(W.shape), W.data)
                if b is not None:
                    b.retain_data()
                    _add_blob(layer, list(b.shape), b.data)

        elif func.label in ('Convolution2DFunction',
                            'Deconvolution2DFunction'):
            if len(func.inputs) == 2:
                _, W = func.inputs
                b = None
            else:
                _, W, b = func.inputs
            n_out, n_in, kw, kh = W.shape
            convolution_param = {
                'num_output': n_out,
                'bias_term': b is not None,
                'pad_w': func.pw,
                'pad_h': func.ph,
                'stride_w': func.sx,
                'stride_h': func.sy,
                'kernel_w': kw,
                'kernel_h': kh,
            }

            params['bottom'] = params['bottom'][:1]
            if func.label == 'Convolution2DFunction':
                params['type'] = 'Convolution'
            else:
                params['type'] = 'Deconvolution'
            params['convolution_param'] = convolution_param

            if net is not None:
                for k, v in six.iteritems(convolution_param):
                    setattr(layer.convolution_param, k, v)
                # print(params['name'], len(W.data.flatten()),
                #       len(b.data.flatten()))
                _add_blob(layer, [n_in, n_out, kh, kw], W.data)

                if b is not None:
                    b.retain_data()
                    _add_blob(layer, [1, n_out], b.data)

        elif func.label in ('MaxPooling2D', 'AveragePooling2D'):
            kw = func.kw
            kh = func.kh
            pooling_param = {
                'pool': 0 if func.label == 'MaxPooling2D' else 1,
                'pad_w': func.pw,
                'pad_h': func.ph,
                'stride_w': func.sx,
                'stride_h': func.sy,
                'kernel_w': kw,
                'kernel_h': kh,
            }
            params['type'] = 'Pooling'
            params['pooling_param'] = pooling_param
            if net is not None:
                for k, v in six.iteritems(pooling_param):
                    setattr(layer.pooling_param, k, v)

        elif func.label == 'LocalResponseNormalization':
            lrn_param = {
                'norm_region': 0,  # ACROSS_CHANNELS
                'local_size': func.n,
                'k': func.k,
                'alpha': func.alpha * func.n,
                'beta': func.beta,
            }
            params['type'] = 'LRN'
            params['lrn_param'] = lrn_param
            if net is not None:
                for k, v in six.iteritems(lrn_param):
                    setattr(layer.lrn_param, k, v)

        elif func.label == 'FixedBatchNormalization':
            _, _, _, mean, var = func.inputs
            batch_norm_param = {'use_global_stats': True}
            params['type'] = 'BatchNorm'
            params['batch_norm_param'] = batch_norm_param
            if net is not None:
                _add_blob(layer, [mean.data.size], mean.data)
                _add_blob(layer, [var.data.size], var.data)
                _add_blob(layer, [1], numpy.ones((1,), dtype='f'))

        elif func.label == 'ReLU':
            params['type'] = 'ReLU'

        elif func.label == 'Concat':
            axis = func.axis
            concat_param = {'axis': axis}
            params['type'] = 'Concat'
            params['concat_param'] = concat_param
            if net is None:
                for k, v in six.iteritems(concat_param):
                    setattr(layer.concat_param, k, v)

        elif func.label == 'Softmax':
            params['type'] = 'Softmax'

        elif func.label == 'Reshape':
            input_ = func.inputs[0]
            parent = input_.creator
            parent_layer_name = parent_layer_names[0]
            if 'Reshape' in parent_layer_name:
                grandparent = parent.inputs[0].creator
                parent_layer_name = self._get_parent_name(grandparent)
            reshape_param = {'shape': {'dim': list(func.shape)}}
            params['type'] = 'Reshape'
            params['bottom'] = parent_layer_name
            params['reshape_param'] = reshape_param
            if layer is not None:
                dim = reshape_param['shape']['dim']
                layer.reshape_param.shape.dim[:] = dim

        else:
            raise Exception(
                'Cannot convert, name={}, rank={}, label={}, inputs={}'.format(
                    layer_name, func.rank, func.label, parent_layer_names))
        if prototxt is not None:
            prototxt.write(self._gen_layer_prototxt(params))

        if net is not None:
            layer.name = params['name']
            layer.type = params['type']
            layer.bottom[:] = params['bottom']
            layer.top[:] = params['top']
            layer.phase = caffe_pb.TEST

    def _get_dump_list(self, outputs):
        funcs = [var.creator for var in outputs if var.creator is not None]
        # output_blobs = [self._get_layer_name(func) for func in funcs]
        dump_list = []
        while funcs:
            func = funcs.pop(0)
            assert isinstance(func, _function_types)

            dump_list.append(func)

            inputs = func.inputs
            for _input in inputs:
                creator = _input.creator
                if creator is not None and creator not in self.seen_funcs:
                    assert isinstance(creator, _function_types)
                    funcs.append(creator)
                    self.seen_funcs.add(creator)
        return dump_list[::-1]

    def __call__(self, name, inputs, outputs):
        dump_list = self._get_dump_list(outputs)
        f = None
        net = None
        if self.caffemodel is not None:
            net = caffe_pb.NetParameter()
        try:
            if self.prototxt is not None:
                f = open(self.prototxt, 'wt')
                f.write('name: "{}"\n'.format(name))
                assert len(inputs) == 1
                f.write('layer {\n'
                        '  name: "data"\n'
                        '  type: "Input"\n'
                        '  top: "data"\n'
                        '  input_param { shape: {')
                for i in inputs[0].shape:
                    f.write(' dim: ' + str(i))
                f.write(' } }\n'
                        '} \n')
            for i in dump_list:
                self.dump_function_object(i, f, net)
        finally:
            if f is not None:
                f.close()

        if net is not None:
            with open(self.caffemodel, 'wb') as f:
                f.write(net.SerializeToString())
            if self.debug:
                import google.protobuf.text_format
                with open(self.caffemodel + ".txt", 'w') as f:
                    f.write(google.protobuf.text_format.MessageToString(net))


def caffe_export(input, output, directory=None,
                 export_params=True, graph_name='Graph'):
    """Export computational graph as Caffe format.

    Args:
        input (~chainer.Variable):
        output (~chainer.Variable):
        directory (str): The directory used for saving the resulting Caffe
            model. If None, nothing is saved to the disk.
        export_params (bool): If True, this function exports all the parameters
            included in the given model at the same time. If False, the
            exported ONNX model doesn't include any parameter values.
        graph_name (str): A string to be used for the ``name`` field of the
            graph in the exported Caffe model.

    """

    assert isinstance(input, (tuple, list))
    assert isinstance(output, (tuple, list))
    for i in output:
        assert isinstance(i, variable.Variable)

    prototxt = None
    caffemodel = None
    if directory is not None:
        prototxt = os.path.join(directory, 'chainer_model.prototxt')
        if export_params:
            caffemodel = os.path.join(directory, 'chainer_model.caffemodel')
    retriever = _RetrieveAsCaffeModel(prototxt, caffemodel)
    retriever.debug = False
    retriever(graph_name, input, output)
