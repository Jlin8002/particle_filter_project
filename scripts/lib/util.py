from typing import List, TypeVar

from geometry_msgs.msg import Quaternion
from numpy import ndarray, random
from tf.transformations import euler_from_quaternion, quaternion_from_euler

T = TypeVar("T")


def yaw_from_quaternion(q: Quaternion) -> float:
    """
    A helper function that takes a quaternion and returns Euler yaw.
    """

    (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

    return yaw


def yaw_to_quaternion(yaw: float) -> ndarray:
    return quaternion_from_euler(ai=0.0, aj=0.0, ak=yaw)


def draw_weighted_sample(
    choices: List[T], probabilities: List[float], n: int
) -> List[T]:
    """
    Return a random sample of n elements from the set choices with the specified probabilities.

    @param `choices`: The values to sample from represented as a list.

    @param `probabilities`: The probability of selecting each element in choices represented as a list.

    @param `n`: The number of samples.
    """

    return random.default_rng().choice(a=choices, size=n, replace=True, p=probabilities)


def draw_uniform_sample(choices: List[T], n: int) -> List[T]:
    return random.default_rng().choice(a=choices, size=n)
