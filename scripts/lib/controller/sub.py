from typing import Callable, TypeVar

from nav_msgs.msg import OccupancyGrid, Odometry
from rospy_util.controller import Sub  # pyright: reportMissingTypeStubs=false
from sensor_msgs.msg import LaserScan

from lib.turtle_bot import TurtlePose
import lib.turtle_bot as turtle

Msg = TypeVar("Msg")


def odometry(to_msg: Callable[[TurtlePose], Msg]) -> Sub[Odometry, Msg]:
    return Sub(
        topic_name="/odom",
        message_type=Odometry,
        to_msg=lambda odom: to_msg(turtle.from_pose(odom.pose.pose)),
    )


def laser_scan(to_msg: Callable[[LaserScan], Msg]) -> Sub[LaserScan, Msg]:
    return Sub(
        topic_name="/scan",
        message_type=LaserScan,
        to_msg=to_msg,
    )


def occupancy_grid(to_msg: Callable[[OccupancyGrid], Msg]) -> Sub[OccupancyGrid, Msg]:
    return Sub(
        topic_name="/map",
        message_type=OccupancyGrid,
        to_msg=to_msg,
    )
