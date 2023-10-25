from __future__ import annotations

from collections.abc import Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pygame

# color scheme: Gruvbox Dark
BACKGROUND_COLOR = (40, 40, 40)
AXIS_COLOR = (124, 111, 100)
BIG_GRID_COLOR = (80, 73, 69)
SMALL_GRID_COLOR = (60, 56, 54)
FUNCTION_COLORS = [
    (204, 36, 29),    # red
    (214, 93, 14),     # orange
    (215, 153, 33),   # yellow
    (152, 151, 26),   # green
    (104, 157, 106),  # green-blue
    (69, 133, 136),   # blue
    (177, 98, 134),   # purple
]


class Function(ABC):
    last_render: list[np.ndarray[(float, float)]] = None
    color: (int, int, int)

    @abstractmethod
    def render(self, plotter: _Plotter):
        pass


@dataclass
class CartesianFunction(Function):
    func: Callable[[float], float]
    color: (int, int, int)

    def render(self, plotter: _Plotter):
        xs = np.linspace(
            plotter.view_position[0] - plotter.view_size[0],
            plotter.view_position[0] + plotter.view_size[0],
            plotter.resolution[0]
        )

        try:
            ys = self.func(xs)
        except ValueError:
            # fallback if we cannot directly apply the function with an array
            ys = np.vectorize(self.func)(xs)

        points = np.vstack((xs, ys)).T
        self.last_render = []
        last_cut = 0
        for index, p in enumerate(points):
            if not (plotter.view_position[1] - plotter.view_size[1]
                    < p[1]
                    < plotter.view_position[1] + plotter.view_size[1]):

                self.last_render.append(points[last_cut:index+1])
                last_cut = index+1

        self.last_render.append(points[last_cut:])


@dataclass
class PolarFunction(Function):
    func: Callable[[float], float]
    color: (int, int, int)

    turns: float = 1.0

    def render(self, plotter: _Plotter):
        thetas = np.linspace(0, 2*self.turns*np.pi, plotter.resolution[0])

        try:
            rs = self.func(thetas)
        except ValueError:
            # fallback if we cannot directly apply the function with an array
            rs = np.vectorize(self.func)(thetas)

        xs = rs * np.cos(thetas)
        ys = rs * np.sin(thetas)
        points = np.vstack((xs, ys)).T
        self.last_render = []
        last_cut = 0
        for index, p in enumerate(points):
            if not (plotter.view_position[1] - plotter.view_size[1]
                    < p[1]
                    < plotter.view_position[1] + plotter.view_size[1]):

                self.last_render.append(points[last_cut:index+1])
                last_cut = index+1

        self.last_render.append(points[last_cut:])


@dataclass
class ParametricFunction(Function):
    func: Callable[[float], (float, float)]
    color: (int, int, int)

    start: float
    end: float
    step: float

    def render(self, plotter: _Plotter):
        ts = np.arange(self.start, self.end, self.step)

        try:
            xs, ys = self.func(ts)
        except ValueError:
            # fallback if we cannot directly apply the function with an array
            xs, ys = np.vectorize(self.func)(ts)

        points = np.vstack((xs, ys)).T

        self.last_render = []
        last_cut = 0
        for index, p in enumerate(points):
            if not (plotter.view_position[1] - plotter.view_size[1]
                    < p[1]
                    < plotter.view_position[1] + plotter.view_size[1]):

                self.last_render.append(points[last_cut:index+1])
                last_cut = index+1

        self.last_render.append(points[last_cut:])


class _Plotter:
    resolution = np.array([960, 720])
    zoom_step = 1.1

    def _graph_function(self, func: Function):
        if self.view_changed:
            func.render(self)

        to_screen_fact = self.resolution / (2 * self.view_size) * (+1, -1)
        half_resolution = self.resolution / 2
        for batch in func.last_render:
            if batch.shape[0] < 2:
                continue

            batch_on_screen = (batch - self.view_position)*to_screen_fact + half_resolution
            pygame.draw.aalines(self.screen, func.color, False, batch_on_screen)

    def _draw_horizontal_line(self, y: float, color: (int, int, int), width: int = 1):
        y_on_screen = (
            self.resolution[1] / 2) * (1 + (self.view_position[1] - y) / self.view_size[1])

        pygame.draw.line(self.screen, color, (0, y_on_screen),
                         (self.resolution[0], y_on_screen), width)

    def _draw_vertical_line(self, x: float, color: (int, int, int), width: int = 1):
        x_on_screen = (
            self.resolution[0] / 2) * (1 - (self.view_position[0] - x) / self.view_size[0])

        pygame.draw.line(self.screen, color, (x_on_screen, 0),
                         (x_on_screen, self.resolution[1]), width)

    def _draw_axis(self):
        self._draw_vertical_line(0.0, AXIS_COLOR)
        self._draw_horizontal_line(0.0, AXIS_COLOR)

    def _calculate_spacing(self, view: float, screen: float):
        min_extent = 100 * view / screen

        exponent = np.floor(np.log10(abs(min_extent)))
        mantissa = min_extent/10**exponent

        major = 1.0
        for m in (2.0, 5.0, 10.0):
            if m > mantissa:
                major = m * 10**exponent
                minor = major / (4 if m == 2 else 5)
                return major, minor

    def _draw_grid_with_spacing(self, hspacing: float, vspacing: float, color: (float, float, float)):

        screen_left_on_plane = self.view_position[0] - self.view_size[0]
        screen_right_on_plane = self.view_position[0] + self.view_size[0]

        line_x = np.floor(screen_left_on_plane / hspacing) * hspacing
        while line_x <= screen_right_on_plane:
            self._draw_vertical_line(line_x, color)
            line_x += hspacing

        screen_up_on_plane = self.view_position[1] - self.view_size[1]
        screen_down_on_plane = self.view_position[1] + self.view_size[1]

        line_y = np.floor(screen_up_on_plane / vspacing) * vspacing
        while line_y <= screen_down_on_plane:
            self._draw_horizontal_line(line_y, color)
            line_y += vspacing

    def _draw_grid(self):
        big_hspacing, small_hspacing = self._calculate_spacing(self.view_size[0], self.resolution[1])
        big_vspacing, small_vspacing = self._calculate_spacing(self.view_size[1]*self.resolution[0]/self.resolution[1], self.resolution[1])
        self._draw_grid_with_spacing(small_hspacing, small_vspacing, SMALL_GRID_COLOR)
        self._draw_grid_with_spacing(big_vspacing, big_vspacing, BIG_GRID_COLOR)

    def _draw_screen(self):
        self.screen.fill(BACKGROUND_COLOR)

        self._draw_grid()
        self._draw_axis()

        for f in self.funcs:
            self._graph_function(f)

        pygame.display.flip()

    def _to_plane_position(self, screen_pos: np.ndarray):
        return self.view_position + self.view_size * (2 * screen_pos / self.resolution - 1) * (1, -1)

    def _update(self) -> bool:
        mouse_pos = self._to_plane_position(np.array(pygame.mouse.get_pos()))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.WINDOWRESIZED:
                new_resolution = np.array([event.x, event.y])

                self.view_size *= new_resolution/self.resolution
                self.resolution = new_resolution
                self.view_changed = True

            if event.type == pygame.MOUSEWHEEL:
                if event.y == 1:
                    self.view_size /= self.zoom_step
                    self.view_position = (self.view_position - mouse_pos)/self.zoom_step + mouse_pos
                else:
                    self.view_size *= self.zoom_step
                    self.view_position = (self.view_position - mouse_pos)*self.zoom_step + mouse_pos
                self.view_changed = True

            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed(3)[0]:
                    self.view_position += event.rel \
                        / self.resolution \
                        * 2 * self.view_size \
                        * (-1, 1)
                    self.view_changed = True

        self._draw_screen()

        pygame.display.set_caption(
            f"FPS: {round(self.clock.get_fps())}"
        )

        self.view_changed = False
        return True

    # PUBLIC API
    def __init__(self):
        self.view_position = np.zeros(2, dtype=float)
        self.view_size = np.array([4.0, 3.0], dtype=float)

        self.funcs = []
        self.next_available_color = 0

    def plot(self, function: Function):
        self.funcs.append(function)

    def plot_cartesian(
            self,
            func: Callable[[float], float],
            color: (int, int, int) = None
            ):

        if color is None:
            color = FUNCTION_COLORS[self.next_available_color]
            self.next_available_color = (self.next_available_color + 1) % len(FUNCTION_COLORS)

        self.plot(CartesianFunction(
            func=func,
            color=color,
        ))

    def plot_polar(
            self,
            func: Callable[[float], float],
            color: (int, int, int) = None,
            turns: float = 1.0
            ):

        if color is None:
            color = FUNCTION_COLORS[self.next_available_color]
            self.next_available_color = (self.next_available_color + 1) % len(FUNCTION_COLORS)

        self.plot(PolarFunction(
            func=func,
            color=color,
            turns=turns
        ))

    def plot_parametric(
            self,
            func: Callable[[float], (float, float)],
            color: (int, int, int) = None,
            start: int = 0.0,
            end: int = 1.0,
            step: int = 0.01,
            ):

        if color is None:
            color = FUNCTION_COLORS[self.next_available_color]
            self.next_available_color = (self.next_available_color + 1) % len(FUNCTION_COLORS)

        self.plot(ParametricFunction(
            func=func,
            color=color,
            start=start,
            end=end,
            step=step,
        ))

    def show(self):
        pygame.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self.screen = pygame.display.set_mode(self.resolution, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.view_changed = True

        running = True
        while running:
            running = self._update()
            self.dt = self.clock.tick() / 1000

        pygame.quit()


plot = _Plotter()

__all__ = ["plot"]
