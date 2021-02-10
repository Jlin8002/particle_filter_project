"""
Subscriptions for receiving map server and robot sensor messages.
"""

# pyright: reportMissingTypeStubs=false

from typing import Callable, TypeVar

from nav_msgs.msg import OccupancyGrid, Odometry
from rospy_util.controller import Sub
from sensor_msgs.msg import LaserScan

from lib.turtle_pose import TurtlePose
import lib.turtle_pose as tp

Msg = TypeVar("Msg")


def odometry(to_msg: Callable[[TurtlePose], Msg]) -> Sub[Odometry, Msg]:
    """
    Receive odometry sensor messages.
    """
    return Sub(
        topic_name="/odom",
        message_type=Odometry,
        to_msg=lambda odom: to_msg(tp.from_pose(odom.pose.pose)),
    )


def laser_scan(to_msg: Callable[[LaserScan], Msg]) -> Sub[LaserScan, Msg]:
    """
    Receive laser scan sensor messages.
    """
    return Sub(
        topic_name="/scan",
        message_type=LaserScan,
        to_msg=to_msg,
    )


def occupancy_grid(to_msg: Callable[[OccupancyGrid], Msg]) -> Sub[OccupancyGrid, Msg]:
    """
    Receive an occupancy grid of the environment from the map server.
    """
    return Sub(
        topic_name="/map",
        message_type=OccupancyGrid,
        to_msg=to_msg,
    )
