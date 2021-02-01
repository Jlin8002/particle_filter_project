from dataclasses import dataclass

from geometry_msgs.msg import Pose


@dataclass
class Particle:
    pose: Pose
    weight: float
