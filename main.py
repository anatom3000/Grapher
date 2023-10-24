from __future__ import annotations
from plotter import plot as plt
import numpy as np


if __name__ == "__main__":
    #plt.plot(lambda x: np.tan(x))
    #plt.plot(lambda x: -np.tan(x/5+2))
    plt.plot(lambda x: np.exp(np.exp(x)))
    
    plt.plot(lambda x: (x-1)*(x-2)*(x-3))

    plt.show()
