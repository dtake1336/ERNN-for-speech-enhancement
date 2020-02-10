import numpy

from chainer import configuration
from chainer import cuda
from chainer import initializers
from chainer import link
from chainer import variable

from .function import fixed_instance_normalization
from .function import InstanceNormalizationFunction


class InstanceNormalization(link.Link):

    """Instance normalization layer on outputs of convolution functions.
    It is recommended to use this normalization instead of batch normalization
    in generative models of what we call Style Transfer.
    """

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 valid_test=False, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(InstanceNormalization, self).__init__()
        self.valid_test = valid_test
        self.avg_mean = None
        self.avg_var = None
        self.N = 0
        if valid_test:
            self.register_persistent('avg_mean')
            self.register_persistent('avg_var')
            self.register_persistent('N')
        self.decay = decay
        self.eps = eps

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, gamma_=None, beta_=None):
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        elif gamma_ is not None:
            gamma = gamma_
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        elif beta_ is not None:
            beta = beta_
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        decay = self.decay
        if (not configuration.config.train) and self.valid_test:
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = fixed_instance_normalization(
                x, gamma, beta, mean, var, self.eps)
        else:
            func = InstanceNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, decay)
            ret = func(x, gamma, beta)
            self.avg_mean = func.running_mean
            self.avg_var = func.running_var

        return ret
