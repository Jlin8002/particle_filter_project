from dataclasses import replace
from functools import reduce
import math
from typing import List

from nav_msgs.msg import OccupancyGrid
from rospy_util.vector2 import Vector2  # pyright: reportMissingTypeStubs=false
import rospy_util.vector2 as v2
from sensor_msgs.msg import LaserScan

from lib.likelihood_field import LikelihoodField
import lib.likelihood_field as lf
from lib.particle import Particle
import lib.particle as particle
from lib.turtle_bot import TurtlePose
from lib.util import draw_weighted_sample, enumerate_step


def initialize(num_particles: int, map: OccupancyGrid) -> List[Particle]:
    return particle.from_occupancy_grid(map, num_particles)


def normalize(particles: List[Particle]) -> List[Particle]:
    total_weight = sum([p.weight for p in particles])

    if total_weight == 0.0:
        return particles

    return [Particle(pose=p.pose, weight=p.weight / total_weight) for p in particles]


def resample(particles: List[Particle]) -> List[Particle]:
    # TODO: recomputing weight list in multiple functions
    weights = [p.weight for p in particles]

    return draw_weighted_sample(
        choices=particles,
        probabilities=weights,
        n=len(particles),
    )


def estimate_pose(particles: List[Particle]) -> TurtlePose:
    # TODO
    return TurtlePose(position=Vector2(0.0, 0.0), yaw=0.0)


def prob_gaussian(dist: float, sd: float) -> float:
    c = 1.0 / (sd * math.sqrt(2.0 * math.pi))
    prob = c * math.exp((-1.0 * (dist ** 2)) / (2.0 * (sd ** 2)))
    return prob


def update_weight(
    particle: Particle,
    field: LikelihoodField,
    scan: LaserScan,
) -> Particle:
    def one_range(weight: float, angle_deg: int, dist: float) -> float:
        if dist > 3.5:
            return weight

        dir = v2.from_angle(particle.pose.yaw + math.radians(angle_deg))
        ztk = particle.pose.position + v2.scale(dir, dist)

        closest = lf.closest_to_pos(field, ztk)
        prob = prob_gaussian(closest, sd=0.8)

        return weight * prob

    # TODO

    weight = reduce(
        lambda wt, angle_dist: one_range(wt, *angle_dist),
        enumerate_step(
            xs=scan.ranges,
            step=36,
        ),
        1.0,
    )

    return replace(particle, weight=weight)


def update_weights(
    particles: List[Particle],
    field: LikelihoodField,
    scan: LaserScan,
) -> List[Particle]:
    # return [update_weight(p, field, scan) for p in particles]
    return particles


def update_poses(
    particles: List[Particle],
    field: LikelihoodField,
    disp_linear: Vector2,
    disp_angular: float,
) -> List[Particle]:
    return [particle.translate(p, field, disp_linear, disp_angular) for p in particles]
