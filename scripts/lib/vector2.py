"""
Two-dimensional vectors.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import acos, cos, sin, sqrt

from numpy import sign

from geometry_msgs.msg import Point


@dataclass
class Vector2:
    """
    A vector in two-dimensional space.
    """

    x: float
    y: float

    def __eq__(v: Vector2, w: object) -> bool:
        return isinstance(w, Vector2) and equals(v, w)

    def __add__(v: Vector2, w: Vector2) -> Vector2:
        return Vector2(x=v.x + w.x, y=v.y + w.y)

    def __sub__(v: Vector2, w: Vector2) -> Vector2:
        return Vector2(x=v.x - w.x, y=v.y - w.y)


zero: Vector2 = Vector2(x=0.0, y=0.0)

right: Vector2 = Vector2(x=1.0, y=0.0)

up: Vector2 = Vector2(x=0.0, y=1.0)


def from_point(point: Point) -> Vector2:
    return Vector2(x=point.x, y=point.y)


def to_point(v: Vector2) -> Point:
    return Point(x=v.x, y=v.y, z=0.0)


def scale(v: Vector2, k: float) -> Vector2:
    """
    Scale a vector by the given scalar.
    """
    return Vector2(x=k * v.x, y=k * v.y)


def dot(v: Vector2, w: Vector2) -> float:
    """
    Compute the dot product of two vectors.
    """
    return v.x * w.x + v.y * w.y


def normalize(v: Vector2) -> Vector2:
    """
    Normalize a vector to the unit vector.
    """
    return scale(v, 1 / magnitude(v))


def magnitude(v: Vector2) -> float:
    """
    Compute the magnitude of a vector.
    """
    return sqrt(dot(v, v))


def sqr_magnitude(v: Vector2) -> float:
    """
    Compute the squared magnitude of a vector.
    """
    return dot(v, v)


def equals(v: Vector2, w: Vector2, epsilon: float = 1e-6) -> bool:
    """
    Check if two vectors are (approximately) equal.

    @param `epsilon`: Maximum difference in the squared magnitude of the vectors
    to consider them equal.
    """
    return sqr_magnitude(v - w) < epsilon


def angle_between(v: Vector2, w: Vector2) -> float:
    """
    Compute the unsigned angle, in the range [0, pi] radians, between two vectors.
    """
    return acos(dot(v, w) / (magnitude(v) * magnitude(w)))


def signed_angle_between(v: Vector2, w: Vector2) -> float:
    """
    Compute the signed angle, in the range [-pi, pi] radians, between two vectors.
    """
    unsigned_angle = angle_between(v, w)
    sn = sign(v.x * w.y - v.y * w.x)
    return sn * unsigned_angle


def distance_between(v: Vector2, w: Vector2) -> float:
    """
    Compute the distance between two vectors.
    """
    return magnitude(v - w)


def from_angle(angle: float) -> Vector2:
    """
    Create a unit vector from an angle relative to the positive x-axis.
    """
    return Vector2(cos(angle), sin(angle))
