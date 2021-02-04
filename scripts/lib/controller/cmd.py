from typing import Any, List

import rospy
from rospy_util.controller import Cmd  # pyright: reportMissingTypeStubs=false

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Header

from lib.particle import Particle
import lib.turtle_bot as turtle
from lib.turtle_bot import TurtlePose

none: List[Cmd[Any]] = []


def mk_header(frame_id: str) -> Header:
    return Header(stamp=rospy.Time.now(), frame_id=frame_id)


def estimated_robot_pose(pose: TurtlePose, frame_id: str) -> Cmd[PoseStamped]:
    header = mk_header(frame_id)
    pose_stamped = PoseStamped(header, turtle.to_pose(pose))

    return Cmd(
        topic_name="/estimated_robot_pose",
        message_type=PoseStamped,
        message_value=pose_stamped,
    )


def particle_cloud(particle_cloud: List[Particle], frame_id: str) -> Cmd[PoseArray]:
    header = mk_header(frame_id)
    poses = [turtle.to_pose(p.pose) for p in particle_cloud]
    pose_array = PoseArray(header, poses)

    return Cmd(
        topic_name="/particle_cloud",
        message_type=PoseArray,
        message_value=pose_array,
    )
