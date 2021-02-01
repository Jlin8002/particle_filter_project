#!/usr/bin/env python3

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, TypeVar

import numpy as np
from numpy.random import random_sample

import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion

T = TypeVar("T")


def get_yaw_from_pose(p: Pose) -> float:
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = euler_from_quaternion(
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    )[2]

    return yaw


def draw_random_sample(choices: List[T], probabilities: List[float], n: int) -> List[T]:
    """Return a random sample of n elements from the set choices with the specified probabilities
    choices: the values to sample from represented as a list
    probabilities: the probability of selecting each element in choices represented as a list
    n: the number of samples
    """
    values = np.array(range(len(choices)))
    probs = np.array(probabilities)
    bins = np.add.accumulate(probs)
    inds = values[np.digitize(random_sample(n), bins)]
    samples = []
    for i in inds:
        samples.append(deepcopy(choices[int(i)]))
    return samples


@dataclass
class Particle:
    # particle pose (Pose object from geometry_msgs)
    pose: Pose

    # particle weight
    w: float


class ParticleFilter:
    def __init__(self) -> None:

        # once everything is setup initialized will be set to true
        self.initialized: bool = False

        # initialize this particle filter node
        rospy.init_node("turtlebot3_particle_filter")

        # set the topic names and frame names
        self.base_frame: str = "base_footprint"
        self.map_topic: str = "map"
        self.odom_frame: str = "odom"
        self.scan_topic: str = "scan"

        # inialize our map and occupancy field
        self.map: OccupancyGrid = OccupancyGrid()

        # the number of particles used in the particle filter
        self.num_particles: int = 10000

        # initialize the particle cloud array
        self.particle_cloud: List[Particle] = []

        # initialize the estimated robot pose
        self.robot_estimate: Pose = Pose()

        # set threshold values for linear and angular movement before we preform an update
        self.lin_mvmt_threshold: float = 0.2
        self.ang_mvmt_threshold: float = np.pi / 6

        self.odom_pose_last_motion_update: Optional[PoseStamped] = None

        # Setup publishers and subscribers

        # publish the current particle cloud
        self.particles_pub: rospy.Publisher = rospy.Publisher(
            "particle_cloud", PoseArray, queue_size=10
        )

        # publish the estimated robot pose
        self.robot_estimate_pub: rospy.Publisher = rospy.Publisher(
            "estimated_robot_pose", PoseStamped, queue_size=10
        )

        # subscribe to the map server
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.get_map)

        # subscribe to the lidar scan from the robot
        rospy.Subscriber(self.scan_topic, LaserScan, self.robot_scan_received)

        # enable listening for and broadcasting corodinate transforms
        self.tf_listener: TransformListener = TransformListener()
        self.tf_broadcaster: TransformBroadcaster = TransformBroadcaster()

        # initialize the particle cloud
        self.initialize_particle_cloud()

        self.initialized: bool = True

    def get_map(self, data: OccupancyGrid) -> None:
        self.map = data

    def initialize_particle_cloud(self) -> None:

        # TODO

        self.normalize_particles()

        self.publish_particle_cloud()

    def normalize_particles(self) -> None:
        # make all the particle weights sum to 1.0
        # TODO

        pass

    def publish_particle_cloud(self) -> None:

        particle_cloud_pose_array: PoseArray = PoseArray()
        particle_cloud_pose_array.header = Header(
            stamp=rospy.Time.now(), frame_id=self.map_topic
        )

        for part in self.particle_cloud:
            particle_cloud_pose_array.poses.append(part.pose)

        self.particles_pub.publish(particle_cloud_pose_array)

    def publish_estimated_robot_pose(self):

        robot_pose_estimate_stamped = PoseStamped()
        robot_pose_estimate_stamped.pose = self.robot_estimate
        robot_pose_estimate_stamped.header = Header(
            stamp=rospy.Time.now(), frame_id=self.map_topic
        )
        self.robot_estimate_pub.publish(robot_pose_estimate_stamped)

    def resample_particles(self):
        # TODO

        pass

    def robot_scan_received(self, data: LaserScan):

        # wait until initialization is complete
        if not (self.initialized):
            return

        # we need to be able to transfrom the laser frame to the base frame
        if not (
            self.tf_listener.canTransform(
                self.base_frame, data.header.frame_id, data.header.stamp
            )
        ):
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated)
        self.tf_listener.waitForTransform(
            self.base_frame, self.odom_frame, data.header.stamp, rospy.Duration(0.5)
        )
        if not (
            self.tf_listener.canTransform(
                self.base_frame, data.header.frame_id, data.header.stamp
            )
        ):
            return

        # calculate the pose of the laser distance sensor
        p = PoseStamped(
            header=Header(stamp=rospy.Time(0), frame_id=data.header.frame_id)
        )

        self.laser_pose = self.tf_listener.transformPose(self.base_frame, p)

        # determine where the robot thinks it is based on its odometry
        p = PoseStamped(
            header=Header(stamp=data.header.stamp, frame_id=self.base_frame),
            pose=Pose(),
        )

        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)

        # we need to be able to compare the current odom pose to the prior odom pose
        # if there isn't a prior odom pose, set the odom_pose variable to the current pose
        if self.odom_pose_last_motion_update is None:
            self.odom_pose_last_motion_update = self.odom_pose
            return

        if self.particle_cloud:

            # check to see if we've moved far enough to perform an update

            curr_x = self.odom_pose.pose.position.x
            old_x = self.odom_pose_last_motion_update.pose.position.x
            curr_y = self.odom_pose.pose.position.y
            old_y = self.odom_pose_last_motion_update.pose.position.y
            curr_yaw = get_yaw_from_pose(self.odom_pose.pose)
            old_yaw = get_yaw_from_pose(self.odom_pose_last_motion_update.pose)

            if (
                np.abs(curr_x - old_x) > self.lin_mvmt_threshold
                or np.abs(curr_y - old_y) > self.lin_mvmt_threshold
                or np.abs(curr_yaw - old_yaw) > self.ang_mvmt_threshold
            ):

                # This is where the main logic of the particle filter is carried out

                self.update_particles_with_motion_model()

                self.update_particle_weights_with_measurement_model(data)

                self.normalize_particles()

                self.resample_particles()

                self.update_estimated_robot_pose()

                self.publish_particle_cloud()
                self.publish_estimated_robot_pose()

                self.odom_pose_last_motion_update = self.odom_pose

    def update_estimated_robot_pose(self) -> None:
        # based on the particles within the particle cloud, update the robot pose estimate
        # TODO

        pass

    def update_particle_weights_with_measurement_model(self, data: LaserScan):
        # TODO

        pass

    def update_particles_with_motion_model(self):
        # based on the how the robot has moved (calculated from its odometry), we'll  move
        # all of the particles correspondingly

        # TODO

        pass


if __name__ == "__main__":
    pf = ParticleFilter()

    rospy.spin()
