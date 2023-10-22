from __future__ import annotations

import itertools
import random

import scipy.special
import numpy as np

from plotter import plot as plt


def taylor_loop(terms, n_iter=200):
    if isinstance(terms, str):
        terms = seq(terms)

    def f(x):
        y = 0.0
        for i in range(n_iter):
            y += terms[i % len(terms)] * x ** i / scipy.special.factorial(i)

        return y

    return f


def seq(string):
    return [{
                "+": 1.0,
                "0": 0.0,
                "-": -1.0
            }[char] for char in string]


if __name__ == "__main__":
    # exp = taylor_loop("+")  # exp(x)
    # zero = taylor_loop("0")  # y = 0
    # _exp = taylor_loop("-")  # -exp(x)

    for f in itertools.product("+0-", repeat=3):
        plt.plot(taylor_loop(''.join(f)), plotting_accuracy=20)


    plt.plot(lambda x: x**2, plotting_accuracy=20)


    # plt.plot(taylor_loop("+00+"))

    # plt.plot(taylor_loop("--++"), FUNC_TINT)
    # plt.plot(lambda x: np.sqrt(2))

    # plt.plot(f, (255, 0, 0))
    # plt.plot(lambda x: -np.exp(-x))
    plt.show()
