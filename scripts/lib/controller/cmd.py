"""
Commands for publishing particle cloud and pose estimate updates.
"""

# pyright: reportMissingTypeStubs=false

from typing import Any, List

from geometry_msgs.msg import PoseArray, PoseStamped
import rospy
from rospy_util.controller import Cmd
from std_msgs.msg import Header

from lib.particle import Particle
import lib.turtle_pose as tp
from lib.turtle_pose import TurtlePose


def update_estimated_robot_pose(pose: TurtlePose, frame_id: str) -> Cmd[PoseStamped]:
    """
    Update the estimated robot pose.
    """
    header = mk_header(frame_id)
    pose_stamped = PoseStamped(header, tp.to_pose(pose))

    return Cmd(
        topic_name="/estimated_robot_pose",
        message_type=PoseStamped,
        message_value=pose_stamped,
    )


def update_particle_cloud(particles: List[Particle], frame_id: str) -> Cmd[PoseArray]:
    """
    Update the particle cloud after each iteration of the particle filter.
    """
    return publish_particle_cloud(particles, frame_id, latch=False)


def init_particle_cloud(particles: List[Particle], frame_id: str) -> Cmd[PoseArray]:
    """
    Publish the the particle cloud after initialization.
    """
    return publish_particle_cloud(particles, frame_id, latch=True)


def publish_particle_cloud(
    particles: List[Particle],
    frame_id: str,
    latch: bool,
) -> Cmd[PoseArray]:
    """
    Publish the particle cloud, optionally latching to ensure the message is
    received by new subscribers.
    """
    pose_array = pose_array_from_particles(particles, frame_id)

    return Cmd(
        topic_name="/particle_cloud",
        message_type=PoseArray,
        message_value=pose_array,
        latch_publisher=latch,
    )


"""
The no-op command (an empty list of commands).
"""
none: List[Cmd[Any]] = []


def pose_array_from_particles(particles: List[Particle], frame_id: str) -> PoseArray:
    """
    Create a PoseArray message from the particle cloud.
    """
    header = mk_header(frame_id)
    poses = [tp.to_pose(p.pose) for p in particles]
    return PoseArray(header, poses)


def mk_header(frame_id: str) -> Header:
    """
    Create a ROS message header for the current time and specified frame.
    """
    return Header(stamp=rospy.Time.now(), frame_id=frame_id)
