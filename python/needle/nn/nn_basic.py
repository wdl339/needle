"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(self.weight, device=device, dtype=dtype)
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            self.bias = self.bias.transpose()
            self.bias = Parameter(self.bias, device=device, dtype=dtype)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias is not None:
            y += self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, classes = logits.shape
        one_hot_y = init.one_hot(classes, y, device=logits.device)
        z_y = ops.summation(logits * one_hot_y, axes=(1,))
        return (ops.logsumexp(logits, axes=(1,)) - z_y).sum() / batch_size
        ### END YOUR SOLUTION

        # 例：logits = [[1, 2, 3],[4, 5, 6]], y = [0, 1]
        # one_hot_y = [[1, 0, 0],[0, 1, 0]]
        # logits * one_hot_y = [[1, 0, 0],[0, 5, 0]]
        # z_y = [1, 5]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(1, dim, device=device, dtype=dtype)
        self.bias = init.zeros(1, dim, device=device, dtype=dtype)
        self.weight = Parameter(self.weight, device=device, dtype=dtype)
        self.bias = Parameter(self.bias, device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size, classes = x.shape
            mean = (x.sum(axes=(0, )) / batch_size).reshape((1, classes))
            var = (((x - mean.broadcast_to(x.shape)) ** 2).sum(axes=(0, )) / batch_size).reshape((1, classes))
            self.running_mean = self.running_mean *(1 - self.momentum) + mean.reshape(self.running_mean.shape) * self.momentum
            self.running_var = self.running_var *(1 - self.momentum) + var.reshape(self.running_var.shape) * self.momentum
            mean = mean.broadcast_to(x.shape)
            var = var.broadcast_to(x.shape)
        else:
            mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, self.dim)).broadcast_to(x.shape)
        x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        weight = self.weight.broadcast_to(x.shape)
        bias = self.bias.broadcast_to(x.shape)
        return x_hat * weight + bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(1, dim, device=device, dtype=dtype)
        self.bias = init.zeros(1, dim, device=device, dtype=dtype)
        self.weight = Parameter(self.weight, device=device, dtype=dtype)
        self.bias = Parameter(self.bias, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, classes = x.shape
        mean = (x.sum(axes=(1,)) / classes).reshape((batch_size, 1)).broadcast_to(x.shape)
        var = (((x - mean) ** 2).sum(axes=(1,)) / classes).reshape((batch_size, 1)).broadcast_to(x.shape)
        x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5)
        weight = self.weight.broadcast_to(x_hat.shape)
        bias = self.bias.broadcast_to(x_hat.shape)
        res = weight * x_hat + bias
        return res
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * init.randb(*x.shape, p=1 - self.p, device=x.device) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
