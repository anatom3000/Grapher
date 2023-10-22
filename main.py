from __future__ import annotations
from plotter import plot as plt
import numpy as np

if __name__ == "__main__":
    plt.plot(lambda x: 1 / (1 + x*x))
    plt.plot(np.log)

    print(np.arccos(2))

    plt.show()
