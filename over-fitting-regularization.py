import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

np.random.seed(29)


# 目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)


# 多项式
def fit_func(p, x):
    # `np.poly1d([1,2,3])`  生成  $1x^2+2x^1+3x^0$
    f = np.poly1d(p)
    return f(x)


# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


# 十个点
x = np.linspace(0, 1, 10)
# 绘图用
x_points = np.linspace(0, 1, 1000)

y_ = real_func(x)
# 加上正态分布噪音的目标函数的值
y = [np.random.normal(0, 0.1) + _ for _ in y_]


def fitting(M=0):
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    #leastsq
    #x = arg min(sum(func(y)**2,axis=0))
    #         y
    p_lsq = sp.optimize.leastsq(residuals_func, p_init, args=(x, y))
    print("Fitting Parameters:", p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label="real")
    plt.plot(x_points,
             fit_func(p_lsq[0], x_points),
             label=r"fitted curve $M=$" + str(M))
    plt.plot(x, y, "bo", label="noise")
    plt.legend()
    plt.show()
    return p_lsq


for m in [0, 1, 3]:
    fitting(m)
p_lsq_9 = fitting(9)
regularization = 0.0001


def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret,
                    np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
    return ret


# 最小二乘法，加正则化项
p_init = np.random.rand(9 + 1)
p_lsq_regularization = sp.optimize.leastsq(residuals_func_regularization,
                                           p_init,
                                           args=(x, y))

plt.plot(x_points, real_func(x_points), label="real")
plt.plot(x_points,
         fit_func(p_lsq_9[0], x_points),
         label=r"fitted curve $M=$" + str(9))
plt.plot(x_points,
         fit_func(p_lsq_regularization[0], x_points),
         label="regularization")
plt.plot(x, y, "bo", label="noise")
plt.legend()
plt.show()
