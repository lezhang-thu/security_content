import numpy as np

import torch
import load_data

batch_size = 256
train_iter, test_iter = load_data.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
lr = 0.1


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = []
    with torch.no_grad():
        for X, y in data_iter:
            metric.append((accuracy(net(X), y), y.numel()))
    metric = np.asarray(metric)
    return metric[:, 0].sum() / metric[:, 1].sum()


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_epoch(net, train_iter, loss):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        l.sum().backward()
        sgd([W, b], lr, X.shape[0])


def train(net, train_iter, test_iter, loss, num_epochs):
    print(evaluate_accuracy(net, test_iter))
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss)
        test_acc = evaluate_accuracy(net, test_iter)
        print(test_acc)


if __name__ == '__main__':
    num_epochs = 10
    train(net, train_iter, test_iter, cross_entropy, num_epochs)
