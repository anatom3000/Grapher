from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame
from pygame.locals import *

BLACK = (0, 0, 0)
WHITE = (255,) * 3
GRID_TINT = np.array([255, 255, 255])
AXIS_COLOR = 128 / 255 * GRID_TINT
GRID_COLOR = 64 / 255 * GRID_TINT


@dataclass
class _Function:
    func: Callable[[np.ndarray[float]], np.ndarray]
    color: (int, int, int)
    width: int
    accuracy: int

    def __post_init__(self):
        self.func = self.func


@dataclass
class GridConfig:
    def snap_vertical_step(self, view_width: float):

        raw_step = view_width / self.density
        # Calculate the logarithm of the raw step, base 10
        log_raw_step = np.log10(raw_step)

        # Round the logarithm down to the nearest integer
        log_rounded_down = np.floor(log_raw_step)

        # Calculate the exponent to which we will raise 10 to get the snapped step
        exponent = log_rounded_down if log_raw_step > 0 else log_raw_step

        # Calculate the base 10 of the logarithm, which gives us the snapped step
        snapped_step = 10 ** exponent

        # Calculate the difference between the raw step and the snapped step
        difference = raw_step - snapped_step

        # If the difference is greater than or equal to 1/5 of the snapped step, snap to 1/2 of the snapped step
        if difference >= snapped_step / 5:
            snapped_step /= 2

        # If the difference is less than 1/5 of the snapped step, snap to 1/5 of the snapped step
        elif difference >= snapped_step / 10:
            snapped_step /= 5

        # Return the snapped step
        return snapped_step

    snap_horizontal_step = snap_vertical_step

    density: float


class _Plotter:
    pygame.font.init()
    resolution = np.array([960, 720])
    font = pygame.font.Font(pygame.font.get_default_font(), 12)
    zoom_step = 2.0

    grid_config = GridConfig(density=2)

    def __init__(self):
        self.view_position = np.zeros(2, dtype=float)
        self.view_size = np.array([4.0, 3.0], dtype=float)

        self.funcs = []

    def _graph_function(self, func: _Function):
        x = np.linspace(self.view_position[0] - self.view_size[0], self.view_position[0] + self.view_size[0],
                        self.resolution[0] // func.accuracy)
        try:
            y = func.func(x)
        except ValueError:
            # fallback if you can't directly apply the function with an array
            y = np.vectorize(func.func)(x)

        if isinstance(y, float):
            y *= np.vectorize(func.func)(x)

        points = np.vstack((x, y)).T
        to_screen_fact = self.resolution / (2 * self.view_size) * (+1, -1)

        points_on_screen = (points - self.view_position) * to_screen_fact + self.resolution / 2

        pygame.draw.lines(self.screen, func.color, False, points_on_screen, width=func.width)

    def _draw_horizontal_line(self, y: float = 0.0, color: (int, int, int) = GRID_COLOR, width: int = 1):
        y_on_screen = (self.resolution[1] / 2) * (1 + (self.view_position[1] - y) / self.view_size[1])

        pygame.draw.line(self.screen, color, (0, y_on_screen), (self.resolution[0], y_on_screen), width)

    def _draw_vertical_line(self, x: float = 0.0, color: (int, int, int) = GRID_COLOR, width: int = 1):
        x_on_screen = (self.resolution[0] / 2) * (1 - (self.view_position[0] - x) / self.view_size[0])

        pygame.draw.line(self.screen, color, (x_on_screen, 0), (x_on_screen, self.resolution[1]), width)

    def _draw_axis(self):
        x0_on_screen = self.resolution[0] / 2 * (1 - self.view_position[0] / self.view_size[0])
        y0_on_screen = self.resolution[1] / 2 * (1 + self.view_position[1] / self.view_size[1])

        pygame.draw.line(self.screen, AXIS_COLOR, (x0_on_screen, 0), (x0_on_screen, self.resolution[1]))
        pygame.draw.line(self.screen, AXIS_COLOR, (0, y0_on_screen), (self.resolution[0], y0_on_screen))

    def _draw_grid(self):
        screen_width_on_plane = 2 * self.view_size[0]

        spacing = self.grid_config.snap_horizontal_step(screen_width_on_plane / self.grid_config.density)

        self.spacing = spacing

        # spacing = 0.1

        screen_left_on_plane = self.view_position[0] - self.view_size[0]
        screen_right_on_plane = self.view_position[0] + self.view_size[0]

        line_x = np.floor(screen_left_on_plane / spacing) * spacing
        while line_x <= screen_right_on_plane:
            self._draw_vertical_line(line_x, GRID_COLOR)
            line_x += spacing

    def _draw_screen(self):
        self.screen.fill(BLACK)

        self._draw_grid()
        self._draw_axis()

        for f in self.funcs:
            self._graph_function(f)

        pygame.display.flip()

    def _to_plane_position(self, screen_pos: np.ndarray):
        return self.view_position + self.view_size * (2 * screen_pos / self.resolution - 1) * (1, -1)

    def _update(self):
        mouse_pos = self._to_plane_position(np.array(pygame.mouse.get_pos()))

        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            if event.type == WINDOWRESIZED:
                self.resolution = np.array([event.x, event.y])

            if event.type == MOUSEWHEEL:
                if event.y == 1:
                    self.view_size /= self.zoom_step
                    self.view_position = (self.view_position - mouse_pos)/self.zoom_step + mouse_pos
                else:
                    self.view_size *= self.zoom_step
                    self.view_position = (self.view_position - mouse_pos)*self.zoom_step + mouse_pos

            if event.type == MOUSEMOTION:
                if pygame.mouse.get_pressed(3)[0]:
                    self.view_position += event.rel / self.resolution * 2 * self.view_size * (-1, 1)

        self._draw_screen()

        pygame.display.set_caption(f"FPS: {round(self.clock.get_fps())} - {tuple(mouse_pos.round(2))} - Spacing: {self.spacing}")

        return True

    def plot(self,
             func: Callable[[np.ndarray[float]], np.ndarray],
             color: (int, int, int) = WHITE,
             line_width: int = 1,
             plotting_accuracy: int = 1
             ):
        self.funcs.append(_Function(
            func=func,
            color=color,
            width=line_width,
            accuracy=plotting_accuracy
        ))

    def show(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.resolution, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        running = True
        while running:
            running = self._update()
            self.dt = self.clock.tick() / 1000

        pygame.quit()


plot = _Plotter()

__all__ = ["plotter"]
