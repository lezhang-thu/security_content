import torch
from torch import nn
import load_data
from softmax_regression import evaluate_accuracy

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


def train_epoch(net, train_iter, loss, optimizer):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()


def train(net, train_iter, test_iter, loss, num_epochs, optimizer):
    print(evaluate_accuracy(net, test_iter))
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)
        print(test_acc)


train_iter, test_iter = load_data.load_data_fashion_mnist(batch_size)
train(net, train_iter, test_iter, loss, num_epochs, optimizer)
