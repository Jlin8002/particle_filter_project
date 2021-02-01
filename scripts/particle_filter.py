from sensor_msgs.msg import LaserScan

from lib.particle_filter_base import ParticleFilterBase


class ParticleFilter(ParticleFilterBase):
    def initialize_particle_cloud(self) -> None:
        # TODO
        pass

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
