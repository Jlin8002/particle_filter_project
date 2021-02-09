# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass

from geometry_msgs.msg import Pose
from rospy_util.vector2 import Vector2
import rospy_util.vector2 as v2

from lib.util import yaw_from_quaternion, yaw_to_quaternion


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
