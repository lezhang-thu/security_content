import torch


def f(sigmoid_activation=False):
    M = torch.normal(0, 1, size=(4, 4))
    x = torch.diag(torch.full((4,), 0.1))
    print('a single matrix \n{}'.format(M))
    for i in range(100):
        if sigmoid_activation:
            M = M @ (x @ torch.normal(0, 1, size=(4, 4)))
        else:
            M = M @ torch.normal(0, 1, size=(4, 4))
    print('after multiplying 100 matrices\n{}'.format(M))


if __name__ == '__main__':
    #f()
    print('*' * 20)
    f(True)
