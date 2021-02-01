from sensor_msgs.msg import LaserScan

import lib.particle as particle
from lib.particle_filter_base import ParticleFilterBase


class ParticleFilter(ParticleFilterBase):
    def initialize_particle_cloud(self) -> None:
        while self.map is None:
            pass

        self.particle_cloud = particle.from_occupancy_grid(
            self.map,
            self.num_particles,
        )

    def normalize_particles(self) -> None:
        # TODO
        pass

    def resample_particles(self) -> None:
        # TODO
        pass

    def update_estimated_robot_pose(self) -> None:
        # TODO
        pass

    def update_particle_weights_with_measurement_model(self, data: LaserScan) -> None:
        # TODO
        pass

    def update_particles_with_motion_model(self):
        # TODO
        pass
