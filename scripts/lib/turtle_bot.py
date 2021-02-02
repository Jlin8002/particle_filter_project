from dataclasses import dataclass

from geometry_msgs.msg import Pose

from lib.util import yaw_from_quaternion, yaw_to_quaternion
from lib.vector2 import Vector2
import lib.vector2 as v2


@dataclass
class TurtlePose:
    position: Vector2
    yaw: float


def from_pose(pose: Pose) -> TurtlePose:
    return TurtlePose(
        position=v2.from_point(pose.position),
        yaw=yaw_from_quaternion(pose.orientation),
    )


def to_pose(pose: TurtlePose) -> Pose:
    return Pose(
        position=v2.to_point(pose.position),
        orientation=yaw_to_quaternion(pose.yaw),
    )
