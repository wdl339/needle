from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(axis=self.axes, keepdims=True)
        self.max_Z = max_Z
        if BACKEND == "np":
            res = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=self.axes)) + max_Z.squeeze()
            return res
        else:
            res = array_api.log(array_api.sum(array_api.exp(Z - max_Z.broadcast_to(Z.shape)), axis=self.axes))
            return res + max_Z.reshape(res.shape)
        ### END YOUR SOLUTION

    # 例：对二维数组[[1, 2, 3],[4, 5, 6]], max_z = [[3],[6]]
    # exp(Z - max_Z) = [[exp(-2), exp(-1), 1],[exp(-2), exp(-1), 1]]
    # log(sum(exp(Z - max_Z))) = log([exp(-2) + exp(-1) + 1, exp(-2) + exp(-1) + 1]) = [0.4076, 0.4076]
    # max_Z.squeeze() = [3, 6]
    # squeeze()的作用是去掉维度为1的维度
    # log(sum(exp(Z - max_Z))) + max_Z.squeeze() = [3.4076, 6.4076]

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        if self.axes is None:
            axes = tuple(range(len(z.shape)))
        else:
            axes = self.axes
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        grad = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return grad * out_grad.reshape(shape).broadcast_to(z.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

