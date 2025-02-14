import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os
import gc
from needle.autograd import Tensor

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = nn.SoftmaxLoss()
    error_rate = 0
    loss = 0
    for x,y in dataloader:
        x = ndl.Tensor(x, device=ndl.cpu())
        y = ndl.Tensor(y, device=ndl.cpu())
        if opt is None:
            model.eval()
        else:
            model.train()
        y_pred = model(x)
        batch_loss = loss_fn(y_pred, y)
        loss += batch_loss.numpy() * x.shape[0]
        if opt is not None:
            opt.reset_grad()
            batch_loss.backward()
            opt.step()
        error_rate += np.sum(np.argmax(y_pred.numpy(), axis=1) != y.numpy())
    return error_rate / len(dataloader.dataset), loss / len(dataloader.dataset)
    ### END YOUR SOLUTION

    # 总结：实例化损失函数 - 从 DataLoader 获取输入 - 模型推理 -
    #       计算损失 - 重置梯度 - 反向传播 - 更新参数 - 计算错误率


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz", data_dir+"/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz", data_dir+"/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model)
        print(f"Epoch {i}: train_error={train_error}, train_loss={train_loss}, test_error={test_error}, test_loss={test_loss}")
        gc.collect()
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION

    # 总结：实例化 Dataset- 实例化 DataLoader- 
    #       实例化模型 - 实例化优化器 - 迭代 epoch


if __name__ == "__main__":
    train_mnist(
        batch_size=150,
        epochs=5,
        optimizer=ndl.optim.SGD,
        lr=0.001,
        weight_decay=0.01,
        hidden_dim=100,
        data_dir="./data"
    )
