#!/usr/bin/env python3

from sensor_msgs.msg import LaserScan

import lib.particle as particle
from lib.particle_filter_base import ParticleFilterBase
from lib.util import *
import numpy as np
from tf.transformations import quaternion_from_euler


class ParticleFilter(ParticleFilterBase):
    def initialize_particle_cloud(self) -> None:
        while self.map is None:
            pass

        self.particle_cloud = particle.from_occupancy_grid(
            self.map,
            self.num_particles,
        )


    def normalize_particles(self) -> None:
        weight_sum = 0

        for p in self.particle_cloud:
            weight_sum += p.weight

        for p in self.particle_cloud:
            p.weight /= weight_sum


    def resample_particles(self) -> None:
        weights = [p.weight for p in self.particle_cloud]

        self.particle_cloud = draw_weighted_sample(
            self.particle_cloud,
            weights,
            self.num_particles)


    def update_estimated_robot_pose(self) -> None:
        pose_array = np.array([[
            p.pose.position.x,
            p.pose.position.y,
            yaw_from_pose(p.pose.orientation),
            p.weight]
            for p in self.particle_cloud])
        
        self.robot_estimate.position.x = np.average(pose_array[0], weights=pose_array[3])
        self.robot_estimate.position.y = np.average(pose_array[1], weights=pose_array[3])
        self.robot_estimate.orientation = quaternion_from_euler(
            0,
            0,
            np.average(pose_array[2], weights=pose_array[3]))


    def update_particle_weights_with_measurement_model(self, data: LaserScan) -> None:
        # TODO
        pass


    def update_particles_with_motion_model(self):
        # TODO
        pass
