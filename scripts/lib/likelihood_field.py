"""
Likelihood field for the measurement model.
"""

# pyright: reportMissingTypeStubs=false

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
    """
    A two-dimensional field of distances from the current cell to the nearest
    occupied cell.

    @attribute `width`: The width of the field in cells.

    @attribute `height`: The height of the field in cells.

    @attribute `resolution`: The side length of each cell in meters.

    @attribute `origin`: The position of the cell (0, 0) in meters.

    @attribute `field`: The 2D field of distances.
    """

    width: int
    height: int
    resolution: float
    origin: Vector2
    field: ndarray


"""
Occupied cells are 0 cells away from themselves.
Unknown cells are encoded using a negative distance.
"""
DIST_OCCUPIED: float = 0.0
DIST_UNKNOWN: float = -1.0


def at_free_pos(field: LikelihoodField, pos: Vector2) -> bool:
    """
    Perform a lookup on the likelihood field at a given position (x, y) in meters
    to check if the position is free.
    """
    if (dist := closest_to_pos(field, pos)) is None:
        return False

    return dist > 0.0


def closest_to_pos(field: LikelihoodField, pos: Vector2) -> Optional[float]:
    """
    Perform a lookup on the likelihood field at a given position (x, y) in meters
    to find the distance in meters to the closest occupied cell.
    """
    row = round((pos.y - field.origin.y) / field.resolution)
    col = round((pos.x - field.origin.x) / field.resolution)

    if (dist := closest_to_index(field, (row, col))) is None:
        return None

    return dist * field.resolution


def closest_to_index(field: LikelihoodField, ix: Tuple[int, int]) -> Optional[float]:
    """
    Perform a lookup on the likelihood field at a given index (row, column) to
    find the distance in cells to the closest occupied cell.
    """
    (row, col) = ix

    if row < 0 or row >= field.height or col < 0 or col >= field.width:
        return None

    if (dist := field.field[row][col]) == DIST_UNKNOWN:
        return None

    return dist


def from_occupancy_grid(grid: OccupancyGrid) -> LikelihoodField:
    """
    Create a likelihood field from the given occupancy grid.
    """

    def to_pos(ix: int) -> Tuple[int, int]:
        """
        Map a 1D occupancy grid index to a 2D index (row, col).
        """
        return (ix % grid.info.width, ix // grid.info.width)

    cells_indexed = [*enumerate(grid.data)]

    occupied_positions = [to_pos(ix) for (ix, c) in cells_indexed if c == cell.OCCUPIED]

    # Use a two-dimensional k-d tree to quickly find the occupied cell closest
    # to a given free cell.
    occupied_tree = kdtree.create(point_list=occupied_positions)

    def compute_distance(ix: int, c: int) -> float:
        """
        Compute the distance from the given cell to the closest occupied cell.

        @param `ix`: The 1D index of the given cell.

        @param `c`: The value representation of the given cell.
        """
        if c == cell.FREE:
            nearest_occupied: Optional[
                Tuple[kdtree.Node, float]
            ] = occupied_tree.search_nn(to_pos(ix), dist=points_dist)

            # Contingency for a map with no occupied cells.
            if nearest_occupied is None:
                return DIST_UNKNOWN

            (_, distance) = nearest_occupied

            return distance

        return DIST_OCCUPIED if c == cell.OCCUPIED else DIST_UNKNOWN

    field_1d = [compute_distance(*indexed) for indexed in cells_indexed]
    field = np.reshape(field_1d, (grid.info.width, grid.info.height))

    return LikelihoodField(
        width=grid.info.width,
        height=grid.info.height,
        resolution=grid.info.resolution,
        origin=v2.from_point(grid.info.origin.position),
        field=field,
    )
