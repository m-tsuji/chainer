import numpy

from six import moves

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import conv
from chainer.utils import conv_nd
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn


class ConvolutionND(function.Function):

    def __init__(self, ndim, stride=1, pad=0, cover_all=False):
        self.ndim = ndim
        self.stride = conv_nd.as_tuple(stride, ndim)
        self.pad = conv_nd.as_tuple(pad, ndim)
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == self.ndim + 2,
            w_type.ndim == self.ndim + 2,
            x_type.shape[1] == w_type.shape[1],
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def _use_cudnn(self, x, W):
        return (not self.cover_all and
                chainer.should_use_cudnn('>=auto') and
                self.ndim > 1 and x.dtype == W.dtype)

    def _forward_xp(self, x, W, b, xp):
        ndim = self.ndim
        ksize = W.shape[2:]
        stride = self.stride
        pad = self.pad

        # Make patch array.
        if xp is numpy:
            self.col = conv_nd.im2col_nd_cpu(
                x, ksize, stride, pad, cover_all=self.cover_all)
        else:
            self.col = conv_nd.im2col_nd_gpu(
                x, ksize, stride, pad, cover_all=self.cover_all)

        # Compute correlation.
        axes = tuple(moves.range(1, ndim + 2))  # (1, 2, ..., N+1)
        y = xp.tensordot(self.col, W, (axes, axes)).astype(x.dtype, copy=False)

        # Apply bias if given.
        if b is not None:
            y += b

        # Roll c_O before the second in (n, y_1, y_2, ..., y_N, c_O).
        return xp.rollaxis(y, ndim + 1, 1),

    def _forward_cudnn(self, x, W, b):
        out_c = W.shape[0]      # (c_O, _, k_1, k_2, ..., k_N)
        ksize = W.shape[2:]
        n, c = x.shape[:2]      # (n, c_I, d_1, d_2, ..., d_N)
        dims = x.shape[2:]
        stride = self.stride
        pad = self.pad

        # Make empty array for result.
        outs = tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p) in zip(dims, ksize, stride, pad))
        assert all(out > 0 for out in outs), 'Output sizes should be positive.'
        y_shape = (n, out_c) + outs  # (n, c_O, out_1, out_2, ..., out_N)
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)
        workspace_size = cuda.get_max_workspace_size()
        dilation = (1,) * self.ndim
        group = 1
        autotune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cudnn.convolution_forward(
            x, W, b, y, pad, stride, dilation, group, workspace_size, autotune,
            tensor_core)
        return y,

    def forward(self, inputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        if not type_check.same_types(*inputs):
            if b is not None:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}, type(b): {2}'
                                 .format(type(W), type(x), type(b)))
            else:
                raise ValueError('numpy and cupy must not be used together\n'
                                 'type(W): {0}, type(x): {1}'
                                 .format(type(W), type(x)))

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            return self._forward_xp(x, W, b, numpy)
        elif not self._use_cudnn(x, W):
            return self._forward_xp(x, W, b, cuda.cupy)
        else:
            return self._forward_cudnn(x, W, b)

    def _backward_xp(self, x, W, b, gy, xp):
        dims = x.shape[2:]     # (n, c_I, d_1, d_2, ..., d_N)
        stride = self.stride
        pad = self.pad
        ndim = self.ndim

        # Compute filter weight gradient.
        # (n, _, out_1, out_2, ..., out_N)
        out_axes = (0,) + tuple(moves.range(2, ndim + 2))
        # (n, _, _, ..., _, out_1, out_2, ..., out_N)
        col_axes = (0,) + tuple(moves.range(ndim + 2, ndim * 2 + 2))

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (xp is numpy and
                not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        gW = xp.tensordot(gy, self.col, (out_axes, col_axes)).astype(
            W.dtype, copy=False)

        # Compute patch array gradient.
        gcol = xp.tensordot(W, gy, (0, 1)).astype(x.dtype, copy=False)
        gcol = xp.rollaxis(gcol, ndim + 1)

        # Compute input gradient.
        if xp is numpy:
            gx = conv_nd.col2im_nd_cpu(gcol, stride, pad, dims)
        else:
            gx = conv_nd.col2im_nd_gpu(gcol, stride, pad, dims)

        # Compute bias gradient if given and return gradients.
        if b is None:
            return gx, gW
        else:
            # (n, _, out_1, out_2, ..., out_N)
            axis = (0,) + tuple(moves.range(2, ndim + 2))
            gb = gy.sum(axis=axis)
            return gx, gW, gb

    def _backward_cudnn(self, x, W, b, gy):
        # Convert to C-contiguous arrays.
        gy = cuda.cupy.ascontiguousarray(gy)

        # Make empty arrays for result.
        gx = cuda.cupy.empty_like(x)
        gW = cuda.cupy.empty_like(W)

        # Compute
        pad = self.pad
        stride = self.stride
        dilation = (1,) * self.ndim
        group = 1
        workspace_size = cuda.get_max_workspace_size()
        deterministic = configuration.config.cudnn_deterministic
        autotune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cudnn.convolution_backward_filter(
            x, gy, gW, pad, stride, dilation, group, workspace_size,
            deterministic, autotune, tensor_core)
        cudnn.convolution_backward_data(
            W, gy, None, gx, pad, stride, dilation, group, workspace_size,
            deterministic, autotune, tensor_core)

        # Compute bias gradient if given and return gradients.
        if b is None:
            return gx, gW
        else:
            gb = cuda.cupy.empty_like(b)
            cudnn.convolution_backward_bias(gy, gb)
            return gx, gW, gb

    def backward(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None

        gy = grad_outputs[0]    # (n, c_O, out_1, out_2, ..., out_N)

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            return self._backward_xp(x, W, b, gy, numpy)
        elif not self._use_cudnn(x, W):
            return self._backward_xp(x, W, b, gy, cuda.cupy)
        else:
            return self._backward_cudnn(x, W, b, gy)


def convolution_nd(x, W, b=None, stride=1, pad=0, cover_all=False):
    """N-dimensional convolution function.

    This is an implementation of N-dimensional convolution which is generalized
    two-dimensional convolution in ConvNets. It takes three variables: the
    input ``x``, the filter weight ``W`` and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`N` is the number of spatial dimensions.
    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`d_1, d_2, ..., d_N` are the size of each axis of the input's
      spatial dimensions, respectively.
    - :math:`k_1, k_2, ..., k_N` are the size of each axis of the filters,
      respectively.
    - :math:`l_1, l_2, ..., l_N` are the size of each axis of the output's
      spatial dimensions, respectively.
    - :math:`p_1, p_2, ..., p_N` are the size of each axis of the spatial
      padding size, respectively.

    Then the ``convolution_nd`` function computes correlations between filters
    and patches of size :math:`(k_1, k_2, ..., k_N)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded tensors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``(-p_1, -p_2, ..., -p_N)`` for each spatial axis.

    Let :math:`(s_1, s_2, ..., s_N)` be the stride of filter application.
    Then, the output size :math:`(l_1, l_2, ..., l_N)` is determined by the
    following equations:

    .. math::

       l_n = (d_n + 2p_n - k_n) / s_n + 1 \\ \\ (n = 1, ..., N)

    If ``cover_all`` option is ``True``, the filter will cover the all
    spatial locations. So, if the last stride of filter does not cover the
    end of spatial locations, an addtional stride will be applied to the end
    part of spatial locations. In this case, the output size is determined by
    the following equations:

    .. math::

       l_n = (d_n + 2p_n - k_n + s_n - 1) / s_n + 1 \\ \\ (n = 1, ..., N)

    The N-dimensional convolution function is defined as follows.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable of shape :math:`(n, c_I, d_1, d_2, ..., d_N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Weight variable of shape :math:`(c_O, c_I, k_1, k_2, ..., k_N)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            One-dimensional bias variable with length :math:`c_O` (optional).
        stride (:class:`int` or :class:`tuple` of :class:`int` s):
            Stride of filter applications :math:`(s_1, s_2, ..., s_N)`.
            ``stride=s`` is equivalent to ``(s, s, ..., s)``.
        pad (:class:`int` or :class:`tuple` of :class:`int` s):
            Spatial padding width for input arrays
            :math:`(p_1, p_2, ..., p_N)`. ``pad=p`` is equivalent to
            ``(p, p, ..., p)``.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            `cover_all` needs to be ``False`` if you want to use cuDNN.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, l_1, l_2, ..., l_N)`.

    .. note::

        This function uses cuDNN implementation for its forward and backward
        computation if ALL of the following conditions are satisfied:

        - ``cuda.cudnn_enabled`` is ``True``
        - ``chainer.config.use_cudnn`` is ``'always'`` or ``'auto'``
        - The number of spatial dimensions is more than one.
        - ``cover_all`` is ``False``
        - The input's ``dtype`` is equal to the filter weight's.
        - The ``dtype`` is FP16, FP32 or FP64. (FP16 is only available when
          cuDNN version :math:`\\geq` v3.)

    Convolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    .. seealso:: :class:`~chainer.links.ConvolutionND`, :func:`convolution_2d`

    .. admonition:: Example

        >>> n = 10
        >>> c_i, c_o = 3, 1
        >>> d1, d2, d3 = 30, 40, 50
        >>> k1, k2, k3 = 10, 10, 10
        >>> p1, p2, p3 = 5, 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, d1, d2, d3)).astype('f')
        >>> x.shape
        (10, 3, 30, 40, 50)
        >>> W = np.random.uniform(0, 1, (c_o, c_i, k1, k2, k3)).astype('f')
        >>> W.shape
        (1, 3, 10, 10, 10)
        >>> b = np.random.uniform(0, 1, (c_o)).astype('f')
        >>> b.shape
        (1,)
        >>> s1, s2, s3 = 2, 4, 6
        >>> y = F.convolution_nd(x, W, b, stride=(s1, s2, s3),\
 pad=(p1, p2, p3))
        >>> y.shape
        (10, 1, 16, 11, 9)
        >>> l1 = int((d1 + 2 * p1 - k1) / s1 + 1)
        >>> l2 = int((d2 + 2 * p2 - k2) / s2 + 1)
        >>> l3 = int((d3 + 2 * p3 - k3) / s3 + 1)
        >>> y.shape == (n, c_o, l1, l2, l3)
        True
        >>> y = F.convolution_nd(x, W, b, stride=(s1, s2, s3),\
 pad=(p1, p2, p3), cover_all=True)
        >>> y.shape == (n, c_o, l1, l2, l3 + 1)
        True

    """
    ndim = len(x.shape[2:])
    func = ConvolutionND(ndim, stride, pad, cover_all)
    if b is None:
        return func(x, W)
    else:
        return func(x, W, b)
