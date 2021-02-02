from dataclasses import dataclass, replace
from enum import Enum
from typing import List

import math
import numpy.random as random

from nav_msgs.msg import OccupancyGrid

from lib.turtle_bot import TurtlePose
from lib.util import draw_uniform_sample
from lib.vector2 import Vector2


@dataclass
class Particle:
    pose: TurtlePose
    weight: float


# http://wiki.ros.org/map_server
class Cell(Enum):
    free = 0
    occupied = 100
    unknown = -1


def translate(particle: Particle, disp_linear: Vector2, disp_angular: float) -> Particle:
    return replace(
        particle,
        pose=TurtlePose(
            position=particle.pose.position + disp_linear,
            yaw=particle.pose.yaw + disp_angular,
        ),
    )


def from_occupancy_grid(grid: OccupancyGrid, num_particles: int) -> List[Particle]:
    cells_free = [ix for (ix, cell) in enumerate(grid.data) if cell == Cell.free]

    _num_particles = min(len(cells_free), num_particles)

    cells_selected = draw_uniform_sample(
        choices=cells_free,
        n=_num_particles,
    )

    rng = random.default_rng()

    def from_cell(index: int) -> Particle:
        col = index % grid.info.width
        row = index // grid.info.width

        position = Vector2(
            x=col * grid.info.resolution,
            y=row * grid.info.resolution,
        )

        yaw = rng.uniform(low=0.0, high=2.0 * math.pi)

        return Particle(
            pose=TurtlePose(position, yaw),
            weight=1.0 / _num_particles,
        )

    particles = [from_cell(c) for c in cells_selected]

    return particles
