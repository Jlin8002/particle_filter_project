"""
Particles for the particle cloud.
"""

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass, replace

import numpy.random as random
from rospy_util.vector2 import Vector2
import rospy_util.vector2 as v2

from lib.likelihood_field import LikelihoodField
import lib.likelihood_field as lf
from lib.turtle_pose import TurtlePose


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
