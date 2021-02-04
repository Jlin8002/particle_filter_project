from typing import List

from nav_msgs.msg import OccupancyGrid
from rospy_util.vector2 import Vector2  # pyright: reportMissingTypeStubs=false
from sensor_msgs.msg import LaserScan

from lib.particle import Particle
import lib.particle as particle
from lib.turtle_bot import TurtlePose
from lib.util import draw_weighted_sample


def initialize(num_particles: int, map: OccupancyGrid) -> List[Particle]:
    return particle.from_occupancy_grid(map, num_particles)


def normalize(particles: List[Particle]) -> List[Particle]:
    total_weight = sum([p.weight for p in particles])

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


def update_weights(particles: List[Particle], scan: LaserScan) -> List[Particle]:
    # TODO
    return particles


def update_poses(
    particles: List[Particle],
    disp_linear: Vector2,
    disp_angular: float,
) -> List[Particle]:
    return [particle.translate(p, disp_linear, disp_angular) for p in particles]
