from __future__ import annotations
from plotter import plot as plt
import numpy as np


if __name__ == "__main__":
    plt.plot_cartesian(lambda x: np.tan(x))
    plt.plot_polar(lambda t: np.log(t+1), turns=8.0)
    plt.plot_polar(lambda t: np.tan(t))
    plt.plot_parametric(lambda t:
                        (np.cosh(t), np.sinh(t)), start=-5.0, end=5.0)

    plt.show()
