#!/usr/bin/env python3

from functools import reduce
from typing import List, Tuple

import numpy as np
import rospy

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from lib.particle import Particle
import lib.particle as particle
from lib.particle_filter_base import ParticleFilterBase
from lib.turtle_bot import TurtlePose
from lib.util import draw_weighted_sample
from lib.vector2 import Vector2

# TODO: iterate over weights once then pass weight list around


class ParticleFilter(ParticleFilterBase):
    def initialize_particle_cloud(
        self,
        num_particles: int,
        occupancy_grid: OccupancyGrid,
    ) -> List[Particle]:
        ps = particle.from_occupancy_grid(occupancy_grid, num_particles)
        print(len(ps))
        return ps

    def normalize_particles(self, particle_cloud: List[Particle]) -> List[Particle]:
        weight_sum = sum([p.weight for p in particle_cloud])

        return [
            Particle(pose=p.pose, weight=p.weight / weight_sum) for p in particle_cloud
        ]

    def resample_particles(self, particle_cloud: List[Particle]) -> List[Particle]:
        weights = [p.weight for p in particle_cloud]

        return draw_weighted_sample(
            choices=particle_cloud,
            probabilities=weights,
            n=len(particle_cloud),
        )

    def update_estimated_robot_pose(self, particle_cloud: List[Particle]) -> TurtlePose:
        def accumulate_components(
            acc: Tuple[List[Tuple[float, float]], List[float], List[float]],
            p: Particle,
        ):
            (ps, ys, ws) = acc
            pos_tup = (p.pose.position.x, p.pose.position.y)

            return ([pos_tup, *ps], [p.pose.yaw, *ys], [p.weight, *ws])

        (positions, yaws, weights) = reduce(
            accumulate_components,
            particle_cloud,
            ([], [], []),
        )

        avg_position = Vector2(
            *np.average(positions, axis=0, weights=weights)
        )  # TODO: which axis ?
        avg_yaw = np.average(yaws, weights=weights)

        return TurtlePose(position=avg_position, yaw=avg_yaw)

    def update_particle_weights_with_measurement_model(
        self,
        particle_cloud: List[Particle],
        laser_scan: LaserScan,
    ) -> List[Particle]:
        # TODO
        return particle_cloud

    def update_particles_with_motion_model(
        self,
        particle_cloud: List[Particle],
        disp_linear: Vector2,
        disp_angular: float,
    ) -> List[Particle]:
        # TODO: Out of bounds check
        return [
            particle.translate(p, disp_linear, disp_angular) for p in particle_cloud
        ]


if __name__ == "__main__":
    ParticleFilter()
    rospy.spin()
