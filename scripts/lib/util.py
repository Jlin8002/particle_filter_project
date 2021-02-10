"""
General utilities.
"""

from functools import reduce
from typing import Any, Callable, List, Tuple, TypeVar

from geometry_msgs.msg import Quaternion
from numpy import random, sqrt
from tf.transformations import euler_from_quaternion, quaternion_from_euler

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def yaw_from_quaternion(q: Quaternion) -> float:
    """
    Extract the yaw, in Euler angles, from a quaternion.
    """

    (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

    return yaw


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Create a quaternion from the given yaw component in Euler angles.
    """
    components = quaternion_from_euler(ai=0.0, aj=0.0, ak=yaw)
    return Quaternion(*components)


def draw_weighted_sample(
    choices: List[T],
    probabilities: List[float],
    n: int,
) -> List[T]:
    """
    Draw a random sample of `n` elements from `choices` with the specified
    `probabilities`.

    @param `choices`: The values from which to sample.

    @param `probabilities`: The probability of selecting each element in
    `choices`.

    @param `n`: The number of elements to draw in the sample.
    """
    return random.default_rng().choice(
        a=choices,
        size=n,
        replace=True,
        p=probabilities,
    )


def draw_uniform_sample(choices: List[T], n: int) -> List[T]:
    """
    Draw a random sample of `n` elements from `choices` with uniform probability.

    @param `choices`: The values from which to sample.

    @param `n`: The number of elements to draw in the sample.
    """
    return random.default_rng().choice(a=choices, size=n)


def points_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute the distance between two points `(x1, y1)`, `(x2, y2)`.
    """
    ((x1, y1), (x2, y2)) = (p1, p2)
    return sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def compose_many(*fs) -> Callable[[Any], Any]:
    """
    Compose any number of (properly typed!) functions from right to left.

    ---
    >>> increment_double_square = compose_many(
            lambda x: x ** 2,
            lambda x: x * 2,
            lambda x: x + 1,
        )

    >>> increment_double_square(1)

    >>> 16
    """
    return reduce(compose, fs, identity)


def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    """
    Compose two functions from right to left.

    ---
    >>> increment_double = compose(
            lambda x: x * 2,
            lambda x: x + 1,
        )

    >>> increment_double(1)

    >>> 4
    """
    return lambda x: f(g(x))


def identity(x: T) -> T:
    """
    The identity function.
    """
    return x
