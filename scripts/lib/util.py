from functools import reduce
from typing import Any, Callable, Iterator, List, Tuple, TypeVar

from geometry_msgs.msg import Quaternion
from numpy import random, sqrt
from tf.transformations import euler_from_quaternion, quaternion_from_euler

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def yaw_from_quaternion(q: Quaternion) -> float:
    """
    A helper function that takes a quaternion and returns Euler yaw.
    """

    (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

    return yaw


def yaw_to_quaternion(yaw: float) -> Quaternion:
    components = quaternion_from_euler(ai=0.0, aj=0.0, ak=yaw)
    return Quaternion(*components)


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


def points_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    ((x1, y1), (x2, y2)) = (p1, p2)
    return sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def compose_many(*fs) -> Callable[[Any], Any]:
    return reduce(compose, fs, identity)


def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    return lambda x: f(g(x))


def identity(x: T) -> T:
    return x


def enumerate_step(
    xs: List[T],
    start: int = 0,
    step: int = 1,
) -> Iterator[Tuple[int, T]]:
    for i in range(start, len(xs), step):
        yield (i, xs[i])
