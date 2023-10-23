from __future__ import annotations
from plotter import plot as plt
import numpy as np


if __name__ == "__main__":
    rand = lambda x: np.random.random(len(x))
    plt.plot(lambda x: x*x+0.2*rand(x))

    plt.show()
