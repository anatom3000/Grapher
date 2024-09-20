from __future__ import annotations
from plotter import plot as plt

# DOCS:
# 
# plot.plot(function: Function)
# 
# plot.cartesian(
#   func: (x: float) -> (y: float),
#   color: (int, int, int) = None,
#   cache: bool = True
# )
#
# plot.polar(
#     func: (Θ: float) -> (r: float),
#     color: (int, int, int) = None,
#     turns: float = 1.0,
#     cache: bool = True
# )

# plot.parametric(
#     func: (t: float) -> (x: float, y: float),
#     color: (int, int, int) = None,
#     start: int = 0.0,
#     end: int = 1.0,
#     step: int = 0.01,
#     cache: bool = True
# )
# 
# plot.show()
# plot.resolution: (int, int) = (960, 720)
# plot.zoom_step: float = 1.1
#
# class Function(ABC):
#     last_render: list[np.ndarray[(float, float)]] = None
#     cache: bool
#     color: (int, int, int)
#
#     @abstractmethod
#     def render(self, plotter: _Plotter):
#         pass


if __name__ == "__main__":
    plt.plot_cartesian(lambda x: x)
    plt.show()
