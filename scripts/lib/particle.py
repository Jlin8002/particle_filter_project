"""
Particles for the particle cloud.
"""

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass, replace
from typing import List

import math
import numpy.random as random
from nav_msgs.msg import OccupancyGrid
from rospy_util.vector2 import Vector2
import rospy_util.vector2 as v2

import lib.cell as cell
from lib.likelihood_field import LikelihoodField
import lib.likelihood_field as lf
from lib.turtle_pose import TurtlePose
from lib.util import draw_uniform_sample, yaw_from_quaternion


@dataclass
class Particle:
    """
    A particle representing a possible pose of the robot during localization.

    @attribute `pose`: The pose (2D position and yaw) of the particle.

    @attribute `weight`: The estimated probability that the particle represents
    the true location of the robot. In the range [0.0, 1.0] when normalized.
    """

    pose: TurtlePose
    weight: float

    def __str__(self) -> str:
        pos = self.pose.position
        yaw = self.pose.yaw
        weight = self.weight

        return f"{{pos: ({pos.x}, {pos.y}), yaw: {yaw}, wt: {weight}}}"


def translate(disp_linear: Vector2, disp_angular: float, p: Particle) -> Particle:
    """
    Translate a particle by the given linear and angular displacements.
    """
    disp_forward = v2.rotate(disp_linear, p.pose.yaw)

    pos_new = p.pose.position + disp_forward
    yaw_new = p.pose.yaw + disp_angular

    pose_new = TurtlePose(pos_new, yaw_new)

    return replace(p, pose=pose_new)


def wiggle(
    noise_linear: float,
    noise_angular: float,
    p: Particle,
) -> Particle:
    """
    Wiggle the pose of a particle by the given linear and angular noise factors.
    """
    pos_new = p.pose.position + Vector2(*random.normal(scale=noise_linear, size=2))
    yaw_new = p.pose.yaw + random.normal(scale=noise_angular)

    pose_new = TurtlePose(pos_new, yaw_new)

    return replace(p, pose=pose_new)


def sanitize(field: LikelihoodField, p: Particle) -> Particle:
    """
    Zero the weight of a particle if it is not within the bounds of the likelihood
    field.
    """
    return p if lf.at_free_pos(field, p.pose.position) else replace(p, weight=0.0)


def from_occupancy_grid(grid: OccupancyGrid, num_particles: int) -> List[Particle]:
    """
    Create a particle cloud of size `num_particles` from the given occupancy grid.
    """
    cells_free = [ix for (ix, c) in enumerate(grid.data) if c == cell.FREE]

    # Use no more than the number of free cells as the number of particles.
    _num_particles = min(len(cells_free), num_particles)

    # Uniformly sample free cells.
    cells_selected = draw_uniform_sample(
        choices=cells_free,
        n=_num_particles,
    )

    rng = random.default_rng()

    origin_pos = v2.from_point(grid.info.origin.position)
    origin_yaw = yaw_from_quaternion(grid.info.origin.orientation)

    def from_cell(index: int) -> Particle:
        # Compute cell column and row using cell index and grid dimensions.
        col = index % grid.info.width
        row = index // grid.info.width

        # Scale cell dimensions by grid resolution to compute position in
        # dimensions of the map.
        pos_relative = Vector2(
            x=col * grid.info.resolution,
            y=row * grid.info.resolution,
        )

        # Uniformly pick a random yaw in range [0.0, 2.0 * pi].
        yaw_relative = rng.uniform(low=0.0, high=2.0 * math.pi)

        # Offset relative position and yaw by origin to obtain absolute values
        # in map space.
        pos_absolute = pos_relative + origin_pos
        yaw_absolute = yaw_relative + origin_yaw

        # Create particle with absolute pose and uniform weights.
        return Particle(
            pose=TurtlePose(pos_absolute, yaw_absolute),
            weight=1.0 / _num_particles,
        )

    # Create a particle for each free cell in those uniformly sampled.
    particles = [from_cell(c) for c in cells_selected]

    return particles
