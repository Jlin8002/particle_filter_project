from dataclasses import dataclass, replace
from functools import partial, reduce
import math
from typing import Callable, List, Optional, Tuple

from nav_msgs.msg import OccupancyGrid
import numpy as np
from rospy_util.vector2 import Vector2  # pyright: reportMissingTypeStubs=false
import rospy_util.vector2 as v2
from sensor_msgs.msg import LaserScan

from lib.likelihood_field import LikelihoodField
import lib.likelihood_field as lf
from lib.particle import Particle
import lib.particle as particle
from lib.turtle_pose import TurtlePose
from lib.util import compose_many, draw_weighted_sample

DIST_MAX: float = 3.5


@dataclass
class Range:
    angle: int
    dist: float


def initialize(num_particles: int, map: OccupancyGrid) -> List[Particle]:
    return particle.from_occupancy_grid(map, num_particles)


def normalize(particles: List[Particle]) -> List[Particle]:
    total_weight = sum([p.weight for p in particles])

    if total_weight == 0.0:
        return []

    return [Particle(pose=p.pose, weight=p.weight / total_weight) for p in particles]


def resample(particles: List[Particle]) -> List[Particle]:
    if not particles:
        return []

    weights = [p.weight for p in particles]

    return draw_weighted_sample(
        choices=particles,
        probabilities=weights,
        n=len(particles),
    )


def estimate_pose(particles: List[Particle]) -> Optional[TurtlePose]:
    def accumulate_components(
        acc: Tuple[List[Tuple[float, float]], List[float], List[float]],
        p: Particle,
    ):
        (ps, ys, ws) = acc
        pos = (p.pose.position.x, p.pose.position.y)
        return ([pos, *ps], [p.pose.yaw, *ys], [p.weight, *ws])

    (positions, yaws, weights) = reduce(
        accumulate_components,
        particles,
        ([], [], []),
    )

    if sum(weights) == 0.0:
        return None

    avg_position = Vector2(*np.average(positions, axis=0, weights=weights))
    avg_yaw = np.average(yaws, weights=weights)

    return TurtlePose(position=avg_position, yaw=avg_yaw)


def prob_gaussian(dist: float, sd: float) -> float:
    c = 1.0 / (sd * math.sqrt(2.0 * math.pi))
    prob = c * math.exp((-1.0 * (dist ** 2)) / (2.0 * (sd ** 2)))
    return prob


def interesting_ranges(ranges: List[float]) -> List[Range]:
    return [
        Range(angle, dist) for (angle, dist) in enumerate(ranges) if dist < DIST_MAX
    ]


def update_weight(
    field: LikelihoodField,
    ranges: List[Range],
    sd: float,
    particle: Particle,
) -> Particle:
    if particle.weight == 0.0:
        return particle

    unknown_factor = 1 / (prob_gaussian(0, sd) ** 2)

    def one_range(weight: float, range: Range, sd: float) -> float:
        if range.dist >= DIST_MAX:
            return weight

        heading = v2.from_angle(particle.pose.yaw + math.radians(range.angle))
        ztk = particle.pose.position + v2.scale(heading, range.dist)

        if (closest := lf.closest_to_pos(field, ztk)) is None:
            return weight * unknown_factor

        prob = prob_gaussian(closest, sd=sd)

        return weight * prob

    weight = reduce(
        lambda wt, range: one_range(wt, range, sd),
        ranges,
        1.0,
    )

    return replace(particle, weight=weight)


def update_poses_and_weights(
    field: LikelihoodField,
    scan: LaserScan,
    disp_linear: Vector2,
    disp_angular: float,
    noise_linear: float,
    noise_angular: float,
    sd_obstacle_dist: float,
    num_ranges: int,
    particles: List[Particle],
) -> List[Particle]:
    up_pose: Callable[[Particle], Particle] = compose_many(
        partial(particle.sanitize, field),
        partial(particle.wiggle, noise_linear, noise_angular),
        partial(particle.translate, disp_linear, disp_angular),
    )

    ranges = interesting_ranges(scan.ranges)
    ranges_to_check = [
        ranges[round(i)]
        for i in np.linspace(0, len(ranges) - 1, num_ranges, endpoint=False)
    ]

    up_weight: Callable[[Particle], Particle] = partial(
        update_weight,
        field,
        ranges_to_check,
        sd_obstacle_dist,
    )

    return [up_weight(up_pose(p)) for p in particles]
