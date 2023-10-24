from __future__ import annotations
from plotter import plot as plt
import numpy as np


if __name__ == "__main__":
    plt.plot(lambda x: np.tan(x/10))
    plt.plot(lambda x: -np.tan(x/5+2))
    plt.plot(np.floor)
    plt.plot(np.exp)

    plt.show()
