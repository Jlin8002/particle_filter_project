from dataclasses import dataclass
from enum import Enum
from typing import List

import math
import numpy.random as random

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

from lib.util import draw_uniform_sample


@dataclass
class Particle:
    pose: Pose
    weight: float


# http://wiki.ros.org/map_server
class Cell(Enum):
    free = 0
    occupied = 100
    unknown = -1


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

        position = Point(
            x=col * grid.info.resolution,
            y=row * grid.info.resolution,
            z=0.0,
        )

        yaw = rng.uniform(low=0.0, high=2.0 * math.pi)

        orientation = quaternion_from_euler(
            ai=0.0,
            aj=0.0,
            ak=yaw,
        )

        return Particle(
            pose=Pose(position, orientation),
            weight=1.0 / _num_particles,
        )

    particles = [from_cell(c) for c in cells_selected]

    return particles
