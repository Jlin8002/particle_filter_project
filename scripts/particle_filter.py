#!/usr/bin/env python3

from functools import reduce
from typing import List, Tuple

import numpy as np

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
    @staticmethod
    def initialize_particle_cloud(
        num_particles: int,
        occupancy_grid: OccupancyGrid,
    ) -> List[Particle]:
        return particle.from_occupancy_grid(occupancy_grid, num_particles)

    @staticmethod
    def normalize_particles(particle_cloud: List[Particle]) -> List[Particle]:
        weight_sum = sum([p.weight for p in particle_cloud])

        return [
            Particle(pose=p.pose, weight=p.weight / weight_sum) for p in particle_cloud
        ]

    @staticmethod
    def resample_particles(particle_cloud: List[Particle]) -> List[Particle]:
        weights = [p.weight for p in particle_cloud]

        return draw_weighted_sample(
            choices=particle_cloud,
            probabilities=weights,
            n=len(particle_cloud),
        )

    @staticmethod
    def update_estimated_robot_pose(particle_cloud: List[Particle]) -> TurtlePose:
        def accumulate_components(
            acc: Tuple[List[Vector2], List[float], List[float]],
            p: Particle,
        ):
            (ps, ys, ws) = acc
            return ([p.pose.position, *ps], [p.pose.yaw, *ys], [p.weight, *ws])

        (positions, yaws, weights) = reduce(
            accumulate_components,
            particle_cloud,
            initial=([], [], []),
        )

        avg_position = np.average(positions, weights=weights)
        avg_yaw = np.average(yaws, weights=weights)

        return TurtlePose(position=avg_position, yaw=avg_yaw)

    @staticmethod
    def update_particle_weights_with_measurement_model(
        particle_cloud: List[Particle],
        laser_scan: LaserScan,
    ) -> List[Particle]:
        # TODO
        pass

    @staticmethod
    def update_particles_with_motion_model(
        particle_cloud: List[Particle],
        disp_linear: Vector2,
        disp_angular: float,
    ) -> List[Particle]:
        # TODO: Out of bounds check (probably create a 2d OccupancyMatrix datatype for convenience)
        return [
            particle.translate(p, disp_linear, disp_angular) for p in particle_cloud
        ]
