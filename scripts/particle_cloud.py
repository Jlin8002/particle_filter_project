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
    sd: float=0.1,
) -> Particle:
    def one_range(weight: float, angle_deg: int, dist: float, sd: float) -> float:
        if dist > 3.5:
            #dist = 3.5
            print("3.5: ", weight)
            return weight

        heading = v2.from_angle(particle.pose.yaw + math.radians(angle_deg))
        ztk = particle.pose.position + v2.scale(heading, dist)

        closest = lf.closest_to_pos(field, ztk)
        print("angle: ", angle_deg)
        print("LF: ", closest)
        prob = prob_gaussian(closest, sd=sd)
        
        print("WP: ", weight* prob)
        return weight * prob

    if particle.weight == 0:
        return particle
    print("reducing... ")
    weight = reduce(
        lambda wt, angle_dist: one_range(wt, *angle_dist, sd),
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
    sd: float=0.1,
) -> List[Particle]:
    return [update_weight(p, field, scan, sd) for p in particles]
    # return particles


def update_poses(
    particles: List[Particle],
    field: LikelihoodField,
    disp_linear: Vector2,
    disp_angular: float,
    noise_linear: float = 0.0,
    noise_angular: float = 0.0,
) -> List[Particle]:
    return [
        particle.translate(
            p,
            field,
            disp_linear,
            disp_angular,
            noise_linear,
            noise_angular,
        )
        for p in particles
    ]
