import torch
from torch import nn


def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 验证上述二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# 卷积层
# 卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出
# 实现二维卷积层
class Conv2D(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 图像中目标的边缘检测
# 检测图像中不同颜色的边缘
# 中间四列为黑色，其余像素为白色
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

# 当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零
K = torch.tensor([[1.0, -1.0]])

# 输出`Y`中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘
Y = corr2d(X, K)
print("Y:\n{}".format(Y))

# 卷积核`K`只可以检测垂直边缘
print(corr2d(X.t(), K))

# 学习卷积核
# 学习由`X`生成`Y`的卷积核

# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f"epoch {i+1}, loss {l.sum():.3f}")

# 所学的卷积核的权重张量
print(conv2d.weight.data.reshape((1, 2)))
