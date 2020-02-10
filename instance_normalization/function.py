import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


class InstanceNormalizationFunction(function.Function):

    """Instance Normalization function.

    This is similar to Batch Normalization, however, different
    in that this function does not require running_mean nor running_var
    and mean and variance are calculated for each tensor in mini batch.
    """

    def __init__(self, eps=2e-5, mean=None, var=None,  decay=0.9, valid_test=False):
        self.running_mean = mean
        self.running_var = var
        self.eps = eps
        self.decay = decay
        self.valid_test = valid_test

        self.mean_cache = None

    def check_type_forward(self, in_types):
        n_in = type_check.eval(in_types.size())
        if n_in != 3:
            raise type_check.InvalidType(
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs[:3]
        if configuration.config.train:
            if self.running_mean is None:
                self.running_mean = xp.zeros(x.shape[:2])
                self.running_var = xp.zeros_like(self.running_mean)
            else:
                self.running_mean = xp.array(self.running_mean)
                self.running_var = xp.array(self.running_var)
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis,) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]
        mean_var_expander = (Ellipsis, None, None)

        axis = (2, 3)
        mean = x.mean(axis=axis)
        var = x.var(axis=axis)
        var += self.eps

        if (not configuration.config.train) and self.valid_test:
            mean = self.fixed_mean
            var = self.fixed_var + self.eps

        self.std = xp.sqrt(var, dtype=var.dtype)

        if xp is numpy:
            self.x_hat = _xhat(x, mean, self.std, mean_var_expander)
            y = gamma * self.x_hat + beta
        else:
            self.x_hat, y = cuda.elementwise(
                'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
                '''
                x_hat = (x - mean) / std;
                y = gamma * x_hat + beta;
                ''',
                'in_fwd')(x, mean[mean_var_expander],
                          self.std[mean_var_expander], gamma, beta)

        if configuration.config.train:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)
            self.running_mean *= self.decay
            tmp_ary = (1 - self.decay) * xp.array(mean)
            self.running_mean += tmp_ary
            del tmp_ary
            self.running_var *= self.decay
            tmp_ary = (1 - self.decay) * adjust * xp.array(var)
            self.running_var += tmp_ary
            del tmp_ary
        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma, beta = inputs[:3]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (2, 3)
        gamma_beta_axis = (0, 2, 3)
        mean_var_expander = (Ellipsis, None, None)
        xp = cuda.get_array_module(x)

        gbeta = gy.sum(axis=gamma_beta_axis)
        ggamma = (gy * self.x_hat).sum(axis=gamma_beta_axis)
        if xp is numpy:
            gx = (gamma / self.std)[mean_var_expander] * (
                gy - (self.x_hat * ggamma[mean_var_expander] + gbeta[mean_var_expander]) / m)
        else:
            inv_m = numpy.float32(1) / m
            gx = cuda.elementwise(
                'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, \
                T inv_m',
                'T gx',
                'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                inv_m)',
                'bn_bwd')(gy, self.x_hat, gamma[expander],
                          self.std[mean_var_expander], ggamma[mean_var_expander],
                          gbeta[mean_var_expander], inv_m)
        return gx, ggamma, gbeta


def fixed_instance_normalization(x, gamma, beta, mean, var, eps=2e-5):
    with configuration.using_config('train', False):
        return InstanceNormalizationFunction(eps, None, None)(x, gamma, beta)
