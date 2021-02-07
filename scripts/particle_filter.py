#!/usr/bin/env python3

# pyright: reportMissingTypeStubs=false

from dataclasses import dataclass, replace
import math
from typing import Any, List, Union, Tuple

from nav_msgs.msg import OccupancyGrid
import rospy
from rospy_util.controller import Cmd, Controller, Sub
from rospy_util.vector2 import Vector2
import rospy_util.vector2 as v2
from sensor_msgs.msg import LaserScan

import lib.controller.cmd as cmd
import lib.controller.sub as sub
from lib.likelihood_field import LikelihoodField
import lib.likelihood_field as lf
from lib.particle import Particle
from lib.turtle_bot import TurtlePose
import particle_cloud as pc

### Model ###


@dataclass
class AwaitMap:
    pass


@dataclass
class AwaitPose:
    likelihood_field: LikelihoodField
    particle_cloud: List[Particle]


@dataclass
class Initialized:
    likelihood_field: LikelihoodField
    particle_cloud: List[Particle]
    pose_most_recent: TurtlePose
    pose_last_update: TurtlePose


Model = Union[AwaitMap, AwaitPose, Initialized]

init_model: Model = AwaitMap()

### Events ###


@dataclass
class LoadMap:
    map: OccupancyGrid


@dataclass
class Move:
    pose: TurtlePose


@dataclass
class Scan:
    scan: LaserScan


Msg = Union[LoadMap, Move, Scan]

### Update ###

NUM_PARTICLES: int = 10000

LIN_MVMT_THRESH: float = 0.2
ANG_MVMT_THRESH: float = math.pi / 6.0


def pose_displacement(p1: TurtlePose, p2: TurtlePose) -> Tuple[Vector2, float]:
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
    new_poses = pc.update_poses(particles, field, disp_linear, disp_angular)
    new_weights = pc.update_weights(new_poses, field, scan)
    normalized = pc.normalize(new_weights)
    resampled = pc.resample(normalized)

    return resampled


def update(msg: Msg, model: Model) -> Tuple[Model, List[Cmd[Any]]]:
    if isinstance(model, AwaitMap) and isinstance(msg, LoadMap):
        cloud_init = pc.normalize(
            pc.initialize(num_particles=NUM_PARTICLES, map=msg.map)
        )

        likelihood_field = lf.from_occupancy_grid(msg.map)

        return (
            AwaitPose(
                likelihood_field=likelihood_field,
                particle_cloud=cloud_init,
            ),
            [cmd.particle_cloud(cloud_init, frame_id="map")],
        )

    if isinstance(model, AwaitPose) and isinstance(msg, Move):
        return (
            Initialized(
                likelihood_field=model.likelihood_field,
                particle_cloud=model.particle_cloud,
                pose_most_recent=msg.pose,
                pose_last_update=msg.pose,
            ),
            cmd.none,
        )

    if isinstance(model, Initialized) and isinstance(msg, Move):
        return (replace(model, pose_most_recent=msg.pose), cmd.none)

    if isinstance(model, Initialized) and isinstance(msg, Scan):
        (disp_lin, disp_ang) = pose_displacement(
            p1=model.pose_most_recent,
            p2=model.pose_last_update,
        )

        if v2.magnitude(disp_lin) < LIN_MVMT_THRESH and abs(disp_ang) < ANG_MVMT_THRESH:
            return (model, cmd.none)

        particle_cloud = update_particle_cloud(
            particles=model.particle_cloud,
            field=model.likelihood_field,
            disp_linear=disp_lin,
            disp_angular=disp_ang,
            scan=msg.scan,
        )

        robot_estimate = pc.estimate_pose(particle_cloud)

        new_model = replace(
            model,
            particle_cloud=particle_cloud,
            pose_last_update=model.pose_most_recent,
        )

        return (
            new_model,
            [
                cmd.particle_cloud(particle_cloud, frame_id="map"),
                cmd.estimated_robot_pose(robot_estimate, frame_id="map"),
            ],
        )

    return (model, cmd.none)


### Subscriptions ###


def subscriptions(model: Model) -> List[Sub[Any, Msg]]:
    if isinstance(model, AwaitMap):
        return [sub.occupancy_grid(LoadMap)]

    if isinstance(model, AwaitPose):
        return [sub.odometry(Move)]

    return [sub.odometry(Move), sub.laser_scan(Scan)]


### Run ###


def run() -> None:
    rospy.init_node("turtlebot3_particle_filter")

    Controller.run(
        model=init_model,
        update=update,
        subscriptions=subscriptions,
    )

    rospy.spin()


if __name__ == "__main__":
    run()
