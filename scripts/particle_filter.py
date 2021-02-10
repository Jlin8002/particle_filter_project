#!/usr/bin/env python3

"""
Framework for running the particle filter.
"""

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass, replace
from functools import partial
import math
from typing import Any, List, Tuple, Union

from nav_msgs.msg import OccupancyGrid
import rospy
import rospy_util.vector2 as v2
from sensor_msgs.msg import LaserScan

from lib import LikelihoodField, Particle, TurtlePose, Vector2
from lib.controller import Cmd, Controller, Sub, cmd, sub
import lib.likelihood_field as lf
from lib.util import compose_many

import particle_cloud as pc


### Model ###

# States to represent the operation of the particle filter.


@dataclass
class AwaitMap:
    """
    Wait for the occupancy grid from the map server.
    """


@dataclass
class AwaitPose:
    """
    Wait for the first odometry measurement from the robot.
    """

    likelihood_field: LikelihoodField
    particle_cloud: List[Particle]


@dataclass
class Initialized:
    """
    Begin updating the particle cloud.
    """

    likelihood_field: LikelihoodField
    particle_cloud: List[Particle]
    pose_most_recent: TurtlePose
    pose_last_update: TurtlePose


# Union type for possible robot states.
Model = Union[AwaitMap, AwaitPose, Initialized]

# Start by awaiting the map from the map server.
init_model: Model = AwaitMap()

### Messsages ###

# Messages wrapping items received by ROS subscribers.


@dataclass
class LoadMap:
    """
    Received the occupancy grid from the map server.
    """

    grid: OccupancyGrid


@dataclass
class Move:
    """
    Received an odometry measurement from the robot.
    """

    pose: TurtlePose


@dataclass
class Scan:
    """
    Received a laser scan from the robot.
    """

    scan: LaserScan


# Union type for possible messages.
Msg = Union[LoadMap, Move, Scan]

### Update ###

# The number of particles to simulate in the cloud.
NUM_PARTICLES: int = 10000

# Linear and angular movement thresholds necessary for the particle filter to update.
MVMT_THRESH_LIN: float = 0.2
MVMT_THRESH_ANG: float = math.pi / 6.0

# Factors for adding noise to the movement of the particles.
NOISE_LIN: float = 0.1
NOISE_ANG: float = 0.4

# Standard deviation for computing simulated scan probabilities.
OBSTACLE_DIST_SD: float = 0.15

# Number of laser scan ranges to consider when simulating scans.
NUM_RANGES: int = 6


def pose_displacement(p1: TurtlePose, p2: TurtlePose) -> Tuple[Vector2, float]:
    """
    Calculate the linear and angular displacements between two TurtleBot poses.
    """
    displacement_linear = v2.rotate(p1.position - p2.position, -1.0 * p1.yaw)
    displacement_angular = p1.yaw - p2.yaw
    return (displacement_linear, displacement_angular)


def update_particle_cloud(
    particles: List[Particle],
    field: LikelihoodField,
    disp_linear: Vector2,
    disp_angular: float,
    scan: LaserScan,
) -> List[Particle]:
    """
    Update the poses and weights for the particle cloud and resample particles
    with replacement.
    """
    return compose_many(
        pc.resample,
        pc.normalize,
        partial(
            pc.update_poses_and_weights,
            field,
            scan,
            disp_linear,
            disp_angular,
            NOISE_LIN,
            NOISE_ANG,
            OBSTACLE_DIST_SD,
            NUM_RANGES,
        ),
    )(particles)


def update(msg: Msg, model: Model) -> Tuple[Model, List[Cmd[Any]]]:
    """
    Update the particle filter depending on the message received and the current
    state.
    """

    # The map is received while awaiting the map.
    if isinstance(model, AwaitMap) and isinstance(msg, LoadMap):
        cloud_init = pc.normalize(
            pc.initialize(num_particles=NUM_PARTICLES, grid=msg.grid)
        )

        likelihood_field = lf.from_occupancy_grid(msg.grid)

        # Begin awaiting the first robot pose.
        return (
            AwaitPose(
                likelihood_field=likelihood_field,
                particle_cloud=cloud_init,
            ),
            [cmd.init_particle_cloud(cloud_init, frame_id="map")],
        )

    # An odometry measurement is received while awaiting an initial robot pose.
    if isinstance(model, AwaitPose) and isinstance(msg, Move):
        # Particle filter is initialized; begin updating the particle cloud.
        return (
            Initialized(
                likelihood_field=model.likelihood_field,
                particle_cloud=model.particle_cloud,
                pose_most_recent=msg.pose,
                pose_last_update=msg.pose,
            ),
            cmd.none,
        )

    # Update the most recent robot pose upon receiving an odometry measurement.
    if isinstance(model, Initialized) and isinstance(msg, Move):
        return (replace(model, pose_most_recent=msg.pose), cmd.none)

    # Update the particle cloud upon receiving a distance scan if the robot has
    # moved beyond the specified thresholds since the last update.
    # Then, publish the estimated robot pose and the updated particle cloud.
    if isinstance(model, Initialized) and isinstance(msg, Scan):
        (disp_lin, disp_ang) = pose_displacement(
            p1=model.pose_most_recent,
            p2=model.pose_last_update,
        )

        if v2.magnitude(disp_lin) < MVMT_THRESH_LIN and abs(disp_ang) < MVMT_THRESH_ANG:
            return (model, cmd.none)

        particle_cloud = update_particle_cloud(
            particles=model.particle_cloud,
            field=model.likelihood_field,
            disp_linear=disp_lin,
            disp_angular=disp_ang,
            scan=msg.scan,
        )

        robot_estimate = pc.estimate_pose(particle_cloud)

        # Create model for next iteration of the particle filter.
        new_model = replace(
            model,
            particle_cloud=particle_cloud,
            pose_last_update=model.pose_most_recent,
        )

        # Command for publishing updated particle cloud.
        update_cloud = cmd.update_particle_cloud(particle_cloud, frame_id="map")

        # Command for publishing estimated robot pose, if one was computed.
        try_update_pose = (
            [cmd.update_estimated_robot_pose(robot_estimate, frame_id="map")]
            if robot_estimate is not None
            else cmd.none
        )

        return (new_model, [update_cloud, *try_update_pose])

    # If none of the above conditions held, then do nothing.
    return (model, cmd.none)


### Subscriptions ###


def subscriptions(model: Model) -> List[Sub[Any, Msg]]:
    """
    Subscribe to ROS topics that are necessary for the current particle filter
    state.
    """

    # While awaiting the map, subscribe to the map server.
    if isinstance(model, AwaitMap):
        # While awaiting the map, subscribe to the map server.
        return [sub.occupancy_grid(LoadMap)]

    # While awaiting the first robot pose, subscribe to odometry messages.
    if isinstance(model, AwaitPose):
        # While awaiting the first robot pose, subscribe to movement messages.
        return [sub.odometry(Move)]

    # Once initialized, subscribe to odometry and laser scan messages.
    return [sub.odometry(Move), sub.laser_scan(Scan)]


### Run ###


def run() -> None:
    """
    Instantiate a ROS node and run the particle filter.
    """
    rospy.init_node("turtlebot3_particle_filter")

    Controller.run(
        model=init_model,
        update=update,
        subscriptions=subscriptions,
    )

    rospy.spin()


if __name__ == "__main__":
    run()
