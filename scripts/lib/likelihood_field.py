from dataclasses import dataclass
from typing import Optional, Tuple

import kdtree
from nav_msgs.msg import OccupancyGrid
from numpy import ndarray
import numpy as np
from rospy_util.vector2 import Vector2
import rospy_util.vector2 as v2

import lib.cell as cell
from lib.util import points_dist


@dataclass
class LikelihoodField:
    width: int
    height: int
    resolution: float
    origin: Vector2
    field: ndarray


OCCUPIED: float = 0.0
UNKNOWN: float = -1.0


def at_free_pos(field: LikelihoodField, pos: Vector2) -> bool:
    return closest_to_pos(field, pos) > 0.0


def closest_to_pos(field: LikelihoodField, pos: Vector2) -> float:
    row = round((pos.y - field.origin.y) / field.resolution)
    col = round((pos.x - field.origin.x) / field.resolution)

    return closest_to_index(field, (row, col)) * field.resolution


def closest_to_index(field: LikelihoodField, ix: Tuple[int, int]) -> float:
    (row, col) = ix

    if row < 0 or row >= field.height or col < 0 or col >= field.width:
        return UNKNOWN

    return field.field[row][col]


def from_occupancy_grid(grid: OccupancyGrid) -> LikelihoodField:
    def to_pos(ix: int) -> Tuple[int, int]:
        return (ix % grid.info.width, ix // grid.info.width)

    cells_indexed = [*enumerate(grid.data)]

    occupied_positions = [to_pos(ix) for (ix, c) in cells_indexed if c == cell.OCCUPIED]

    occupied_tree = kdtree.create(point_list=occupied_positions)

    def compute_distance(ix: int, c: int) -> float:
        if c == cell.FREE:
            nearest_occupied: Optional[
                Tuple[kdtree.Node, float]
            ] = occupied_tree.search_nn(to_pos(ix), dist=points_dist)

            if nearest_occupied is None:
                return UNKNOWN

            (_, distance) = nearest_occupied

            return distance

        return OCCUPIED if c == cell.OCCUPIED else UNKNOWN

    field_1d = [compute_distance(*indexed) for indexed in cells_indexed]
    field = np.reshape(field_1d, (grid.info.width, grid.info.height))

    return LikelihoodField(
        width=grid.info.width,
        height=grid.info.height,
        resolution=grid.info.resolution,
        origin=v2.from_point(grid.info.origin.position),
        field=field,
    )
