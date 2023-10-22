from __future__ import annotations
from plotter import plot as plt
import numpy as np

if __name__ == "__main__":
    plt.plot(lambda x: x+1 if x > 0 else x)

    plt.plot(
        lambda x: np.sin(x),
    )

    plt.show()
