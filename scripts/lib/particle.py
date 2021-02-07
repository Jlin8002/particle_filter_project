from dataclasses import dataclass, replace
from typing import List

import math
import numpy as np
import numpy.random as random
from nav_msgs.msg import OccupancyGrid
from rospy_util.vector2 import Vector2  # pyright: reportMissingTypeStubs=false
import rospy_util.vector2 as v2

import lib.cell as cell
from lib.likelihood_field import LikelihoodField
import lib.likelihood_field as lf
from lib.turtle_bot import TurtlePose
from lib.util import draw_uniform_sample, yaw_from_quaternion


@dataclass
class Particle:
    pose: TurtlePose
    weight: float


def translate(
    particle: Particle,
    field: LikelihoodField,
    disp_linear: Vector2,
    disp_angular: float,
) -> Particle:

    def rotate_vector(v: Vector2, theta: float) -> Vector2:
        return Vector2(x=((v.x * math.cos(theta)) + (v.y * -math.sin(theta))),
                       y=((v.x * math.sin(theta)) + (v.y * math.cos(theta))))


    dir_particle = particle.pose.yaw
    disp_forward = rotate_vector(disp_linear, dir_particle)
    pos_new = particle.pose.position + disp_forward

    if not lf.at_free_pos(field, pos_new):
        return replace(particle, weight=0.0)

    yaw_new = particle.pose.yaw + disp_angular
    pose_new = TurtlePose(pos_new, yaw_new)

    return replace(particle, pose=pose_new)


def from_occupancy_grid(grid: OccupancyGrid, num_particles: int) -> List[Particle]:
    cells_free = [ix for (ix, c) in enumerate(grid.data) if c == cell.FREE]  # TODO: eh

    _num_particles = min(len(cells_free), num_particles)

    cells_selected = draw_uniform_sample(
        choices=cells_free,
        n=_num_particles,
    )

    rng = random.default_rng()

    origin_pos = v2.from_point(grid.info.origin.position)
    origin_yaw = yaw_from_quaternion(grid.info.origin.orientation)

    def from_cell(index: int) -> Particle:
        col = index % grid.info.width
        row = index // grid.info.width

        pos_relative = Vector2(
            x=col * grid.info.resolution,
            y=row * grid.info.resolution,
        )

        yaw_relative = rng.uniform(low=0.0, high=2.0 * math.pi)

        pos_absolute = Vector2(2, 1.5) #+ origin_pos # pos_relative + origin_pos
        yaw_absolute = yaw_relative + origin_yaw  # TODO: need to wrap at 2 pi ?

        return Particle(
            pose=TurtlePose(pos_absolute, yaw_absolute),
            weight=1.0 / _num_particles,
        )

    particles = [from_cell(c) for c in cells_selected]

    return particles
