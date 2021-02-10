"""
Implementation of the particle cloud.
"""

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass, replace
from functools import partial, reduce
import math
from typing import Callable, List, Optional, Tuple

from nav_msgs.msg import OccupancyGrid
import numpy as np
import rospy_util.vector2 as v2
from sensor_msgs.msg import LaserScan

from lib import (
    LikelihoodField,
    Particle,
    TurtlePose,
    Vector2,
    cell,
)
import lib.likelihood_field as lf
import lib.particle as particle
from lib.util import (
    compose_many,
    draw_uniform_sample,
    draw_weighted_sample,
    yaw_from_quaternion,
)

"""
Maximum TurtleBot3 laser scan distance.
"""
DIST_MAX: float = 3.5


@dataclass
class Range:
    """
    A single laser scan range with distance below the maximum.
    """

    angle: int
    dist: float


def initialize(num_particles: int, grid: OccupancyGrid) -> List[Particle]:
    """
    Create a particle cloud of size `num_particles` from the given occupancy grid.
    """
    cells_free = [ix for (ix, c) in enumerate(grid.data) if c == cell.FREE]

    # Use no more than the number of free cells as the number of particles.
    _num_particles = min(len(cells_free), num_particles)

    # Uniformly sample free cells.
    cells_selected = draw_uniform_sample(
        choices=cells_free,
        n=_num_particles,
    )

    rng = np.random.default_rng()

    origin_pos = v2.from_point(grid.info.origin.position)
    origin_yaw = yaw_from_quaternion(grid.info.origin.orientation)

    def from_cell(index: int) -> Particle:
        # Compute cell column and row using cell index and grid dimensions.
        col = index % grid.info.width
        row = index // grid.info.width

        # Scale cell dimensions by grid resolution to compute position in
        # dimensions of the map.
        pos_relative = Vector2(
            x=col * grid.info.resolution,
            y=row * grid.info.resolution,
        )

        # Uniformly pick a random yaw in range [0.0, 2.0 * pi].
        yaw_relative = rng.uniform(low=0.0, high=2.0 * math.pi)

        # Offset relative position and yaw by origin to obtain absolute values
        # in map space.
        pos_absolute = pos_relative + origin_pos
        yaw_absolute = yaw_relative + origin_yaw

        # Create particle with absolute pose and uniform weights.
        return Particle(
            pose=TurtlePose(pos_absolute, yaw_absolute),
            weight=1.0 / _num_particles,
        )

    # Create a particle for each free cell in those uniformly sampled.
    particles = [from_cell(c) for c in cells_selected]

    return particles


def normalize(particles: List[Particle]) -> List[Particle]:
    """
    Normalize the weights of the particles in the cloud such that they sum to 1.0.
    If the total weight is 0.0, then return an empty cloud.
    """
    total_weight = sum([p.weight for p in particles])

    if total_weight == 0.0:
        return []

    return [Particle(pose=p.pose, weight=p.weight / total_weight) for p in particles]


def resample(particles: List[Particle]) -> List[Particle]:
    """
    Resample particles from the cloud with replacement based on their weights.
    """
    if not particles:
        return []

    weights = [p.weight for p in particles]

    return draw_weighted_sample(
        choices=particles,
        probabilities=weights,
        n=len(particles),
    )


def estimate_pose(particles: List[Particle]) -> Optional[TurtlePose]:
    """
    Try to estimate the current robot pose from a particle cloud, by calculating
    its centroid and weighted average angle. If all particles have weight 0.0,
    or if there are no particles in the cloud, return None.
    """

    def accumulate_components(
        acc: Tuple[List[Tuple[float, float]], List[float], List[float]],
        p: Particle,
    ):
        """
        Accumulate a particle's position, yaw, and weight into lists.
        """
        (ps, ys, ws) = acc
        pos = (p.pose.position.x, p.pose.position.y)
        return ([pos, *ps], [p.pose.yaw, *ys], [p.weight, *ws])

    (positions, yaws, weights) = reduce(
        accumulate_components,
        particles,
        ([], [], []),
    )

    # Case where no particles are in the cloud (all weight 0).
    if sum(weights) == 0.0:
        return None

    avg_position = Vector2(*np.average(positions, axis=0, weights=weights))
    avg_yaw = np.average(yaws, weights=weights)

    return TurtlePose(position=avg_position, yaw=avg_yaw)


def prob_gaussian(dist: float, sd: float) -> float:
    """
    Compute the value at `dist` of a Gaussian PDF centered on 0, with standard
    deviation `sd` and area 1.
    """
    c = 1.0 / (sd * math.sqrt(2.0 * math.pi))
    prob = c * math.exp((-1.0 * (dist ** 2)) / (2.0 * (sd ** 2)))
    return prob


def valid_ranges(scan: LaserScan) -> List[Range]:
    """
    Extract LaserScan ranges with values below the maximum scan distance.
    """
    return [
        Range(angle, dist)
        for (angle, dist) in enumerate(scan.ranges)
        if dist < DIST_MAX
    ]


def update_weight(
    field: LikelihoodField,
    robot_ranges: List[Range],
    sd_obstacle_dist: float,
    particle: Particle,
) -> Particle:
    """
    Calculate the weight of a single particle based on how well its surroundings
    match the robot's scan values.
    """
    if particle.weight == 0.0:
        return particle

    # Penalty applied to the weight if the particle's simulated scan falls outside
    # the map.
    unknown_factor = 1 / (prob_gaussian(0, sd_obstacle_dist) ** 2)

    def one_range(weight: float, range: Range) -> float:
        """
        Update the particle's weight for a single laser scan range measured by
        the robot.
        """

        # Calculate the endpoint of the particle's simulated scan for the current
        # range.
        heading = v2.from_angle(particle.pose.yaw + math.radians(range.angle))
        ztk = particle.pose.position + v2.scale(heading, range.dist)

        # Apply the penalty for out-of-bounds simulated scans.
        if (closest := lf.closest_to_pos(field, ztk)) is None:
            return weight * unknown_factor

        # Use a Gaussian PDF to penalize simulated scan values that differ
        # significantly from the robot's scan values.
        prob = prob_gaussian(closest, sd=sd_obstacle_dist)

        return weight * prob

    weight = reduce(
        one_range,
        robot_ranges,
        1.0,
    )

    return replace(particle, weight=weight)


def update_poses_and_weights(
    field: LikelihoodField,
    robot_scan: LaserScan,
    disp_linear: Vector2,
    disp_angular: float,
    noise_linear: float,
    noise_angular: float,
    sd_obstacle_dist: float,
    num_ranges: int,
    particles: List[Particle],
) -> List[Particle]:
    """
    Update the poses and weights of all particles in the cloud.

    @param `field`: Likelihood field for computing minimum scan distances and
    verifying updated particle positions.

    @param `robot_scan`: The LaserScan measured by the robot.

    @param: `disp_linear`: The linear displacement of the robot since the last
    particle filter update.

    @param `disp_angular`: The angular displacement of the robot since the last
    particle filter update.

    @param `sd_obstacle_dist`: The standard deviation of the Gaussian used for
    weight probability computation.

    @param: `num_ranges`: The number of robot scan ranges to use for each weight
    computation.

    @param `particles`: The current particle cloud.
    """

    # Function to update the pose of a single particle:
    # 1) Translate the particle by the given linear and angular displacements.
    # 2) Apply noise to (wiggle) the new position and rotation of the particle.
    # 3) Zero the weight of (sanitize) the particle if it is not at a free position
    #    on the map.
    upd_pose: Callable[[Particle], Particle] = compose_many(
        partial(particle.sanitize, field),
        partial(particle.wiggle, noise_linear, noise_angular),
        partial(particle.translate, disp_linear, disp_angular),
    )

    # Pick equally spaced ranges from the laser scan measurements under the maximum
    # distance to use for probability computation when updating weights.
    ranges = valid_ranges(robot_scan)
    ranges_to_check = [
        ranges[round(i)]
        for i in np.linspace(
            start=0,
            stop=len(ranges) - 1,
            num=num_ranges,
            endpoint=False,
        )
    ]

    # Function to update the weight of a single particle.
    upd_weight: Callable[[Particle], Particle] = partial(
        update_weight,
        field,
        ranges_to_check,
        sd_obstacle_dist,
    )

    return [upd_weight(upd_pose(p)) for p in particles]
