import math
from typing import List, Tuple, TypeVar

from geometry_msgs.msg import Point, Pose
from numpy import random
from tf.transformations import euler_from_quaternion

T = TypeVar("T")


def yaw_from_pose(p: Pose) -> float:
    """
    A helper function that takes in a Pose object (geometry_msgs) and returns yaw
    """

    (_, _, yaw) = euler_from_quaternion(
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    )

    return yaw


def draw_random_sample(choices: List[T], probabilities: List[float], n: int) -> List[T]:
    """
    Return a random sample of n elements from the set choices with the specified probabilities.

    @param `choices`: The values to sample from represented as a list.

    @param `probabilities`: The probability of selecting each element in choices represented as a list.

    @param `n`: The number of samples.
    """

    return random.default_rng().choice(a=choices, size=n, replace=True, p=probabilities)


def points_distance(p1: Point, p2: Point) -> float:
    return distance((p1.x, p1.y), (p2.x, p2.y))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    (x1, y1) = p1
    (x2, y2) = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
