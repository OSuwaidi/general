# بسم الله الرحمن الرحيم و به نستعين

import numpy as np
from tqdm import trange
from typing import Callable


def gd(weight_dims: int, loss_f: Callable, gradient: Callable, lr=1, epochs=50, seed=0):
    np.random.seed(seed)
    weight = np.random.randn(weight_dims)
    losses = [loss_f(weight)]

    for _ in trange(epochs):
        g = gradient(weight)
        weight -= g * lr

        losses.append(loss_f(weight))
    return weight, losses


def bouncy_gd(weight_dims: int, loss_f: Callable, gradient: Callable, lr=1, epochs=50, TH=0.7, seed=0, beta=1):
    np.random.seed(seed)
    weight = np.random.randn(weight_dims)
    losses = [loss_f(weight)]
    lr = np.ones(weight_dims)*lr  # alpha
    sw = 1  # v_t

    def dist(g1, g2):
        e = 1e-05
        d1, d2 = np.linalg.norm(g1), np.linalg.norm(g2)
        dists = np.array([d2, d1])
        return dists / (dists.sum() + e)

    for _ in trange(epochs):
        g = gradient(weight)
        sw = beta*sw + abs(g)
        oracle = weight - g * lr
        g_orc = gradient(oracle)
        if g @ g_orc <= 0:
            # print('bounce!')

            d1, d2 = dist(g, g_orc)
            if d1 > TH:  # Implies that we're approaching some minima
                # print("Cut!")
                lr /= sw

            weight = (weight*d1 + oracle*d2)

        else:
            weight = oracle - g_orc * lr

        losses.append(loss_f(weight))

    return weight, losses


def main():
    mem = Memory("./mycache")
    plt.style.use('seaborn')
    plt.rcParams['figure.autolayout'] = True

    @mem.cache
    def get_data(filePath):
        data_set = load_svmlight_file(filePath)
        return data_set[0], data_set[1]

    data, target = get_data("news20.binary.bz2")
    n, d = data.shape
    # data = data.toarray()
    # data = np.append(data, np.ones((n, 1)), 1)
    # d += 1
    print(f"We have {n} samples, each has {d} features")

    def logistic_loss(w):
        return np.mean(np.log(1 + np.exp(-target * (data @ w))))

    def logistic_grad(w):
        e = np.exp(-target * (data @ w))
        gradient = ((-e / (1 + e) * target) @ data) / n
        return gradient

    epochs = 50
    LR = 1000
    _, losses_gd = gd(d, logistic_loss, logistic_grad, lr=LR, epochs=epochs)
    print(f'Loss GD: {losses_gd[-1]:.4}')
    plt.semilogy(losses_gd, label='Vanilla GD')

    _, losses_bgd = bouncy_gd(d, logistic_loss, logistic_grad, lr=LR, epochs=epochs)
    print(f'Loss BGD: {losses_bgd[-1]:.4}')
    plt.semilogy(losses_bgd, label='Bouncing GD')
    plt.ylabel('Loss', rotation='horizontal')
    plt.xlabel('Iterations')
    plt.gca().text(15, 3, f'$LR={LR}$', c='pink', size=20)
    plt.legend(prop={'size': 15})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':  # Such that when you import a function from this script, the whole script doesn't run automatically
    from joblib import Memory
    from sklearn.datasets import load_svmlight_file
    import matplotlib.pyplot as plt

    main()
