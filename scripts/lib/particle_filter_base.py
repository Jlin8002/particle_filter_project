from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import List, Tuple, Union

import rospy

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from tf import TransformBroadcaster, TransformListener

from lib.particle import Particle
from lib.turtle_bot import TurtlePose
import lib.turtle_bot as turtle
from lib.vector2 import Vector2
import lib.vector2 as v2


def pose_displacement(p1: TurtlePose, p2: TurtlePose) -> Tuple[Vector2, float]:
    displacement_linear = p1.position - p2.position
    displacement_angular = abs(p1.yaw - p2.yaw)

    return (displacement_linear, displacement_angular)


@dataclass
class AwaitGrid:
    pass


@dataclass
class AwaitOdom:
    grid: OccupancyGrid


@dataclass
class Initialized:
    particle_cloud: List[Particle]
    pose_previous: TurtlePose


State = Union[AwaitGrid, AwaitOdom, Initialized]


class ParticleFilterBase(ABC):
    # topic names and frame names
    base_frame: str = "base_footprint"
    map_topic: str = "map"
    odom_frame: str = "odom"
    scan_topic: str = "scan"

    # the number of particles used in the particle filter
    num_particles: int = 10000

    # threshold values for linear and angular movement before we preform an update
    lin_mvmt_threshold: float = 0.2
    ang_mvmt_threshold: float = math.pi / 6.0

    def __init__(self) -> None:
        self.state: State = AwaitGrid()

        # initialize this particle filter node
        rospy.init_node("turtlebot3_particle_filter")

        # Setup publishers and subscribers

        # publish the current particle cloud
        self.particles_pub: rospy.Publisher = rospy.Publisher(
            "particle_cloud",
            PoseArray,
            queue_size=10,
        )

        # publish the estimated robot pose
        self.robot_estimate_pub: rospy.Publisher = rospy.Publisher(
            "estimated_robot_pose",
            PoseStamped,
            queue_size=10,
        )

        # subscribe to the map server
        rospy.Subscriber(
            ParticleFilterBase.map_topic,
            OccupancyGrid,
            self.get_grid,
        )

        # subscribe to the lidar scan from the robot
        rospy.Subscriber(
            ParticleFilterBase.scan_topic,
            LaserScan,
            self.robot_scan_received,
        )

        # enable listening for and broadcasting corodinate transforms
        self.tf_listener: TransformListener = TransformListener()
        self.tf_broadcaster: TransformBroadcaster = TransformBroadcaster()

    def get_grid(self, grid: OccupancyGrid) -> None:
        if isinstance(self.state, AwaitGrid):
            self.state = AwaitOdom(grid=grid)

    def publish_particle_cloud(self, particle_cloud: List[Particle]) -> None:

        header = Header(stamp=rospy.Time.now(), frame_id=ParticleFilterBase.map_topic)

        poses = [turtle.to_pose(p.pose) for p in particle_cloud]

        particle_cloud_pose_array = PoseArray(header, poses)

        self.particles_pub.publish(particle_cloud_pose_array)

    def publish_estimated_robot_pose(self, robot_estimate: TurtlePose):

        header = Header(stamp=rospy.Time.now(), frame_id=ParticleFilterBase.map_topic)

        pose = turtle.to_pose(robot_estimate)

        robot_pose_estimate_stamped = PoseStamped(header, pose)

        self.robot_estimate_pub.publish(robot_pose_estimate_stamped)

    def robot_scan_received(self, data: LaserScan) -> None:
        if isinstance(self.state, AwaitGrid):
            return

        # we need to be able to transfrom the laser frame to the base frame
        if not (
            self.tf_listener.canTransform(
                ParticleFilterBase.base_frame,
                data.header.frame_id,
                data.header.stamp,
            )
        ):
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated)
        self.tf_listener.waitForTransform(
            ParticleFilterBase.base_frame,
            ParticleFilterBase.odom_frame,
            data.header.stamp,
            rospy.Duration(0.5),
        )

        if not (
            self.tf_listener.canTransform(
                ParticleFilterBase.base_frame,
                data.header.frame_id,
                data.header.stamp,
            )
        ):
            return

        # determine where the robot thinks it is based on its odometry
        pose_empty = PoseStamped(
            header=Header(
                stamp=data.header.stamp,
                frame_id=ParticleFilterBase.base_frame,
            ),
            pose=Pose(),
        )

        pose_current = turtle.from_pose(
            self.tf_listener.transformPose(
                ParticleFilterBase.odom_frame,
                pose_empty,
            ).pose
        )

        if isinstance(self.state, AwaitOdom):
            particle_cloud = self.normalize_particles(
                particle_cloud=self.initialize_particle_cloud(
                    num_particles=ParticleFilterBase.num_particles,
                    occupancy_grid=self.state.grid,
                ),
            )

            self.state = Initialized(
                particle_cloud=particle_cloud,
                pose_previous=pose_current,
            )

            self.publish_particle_cloud(particle_cloud)

            return

        # check to see if we've moved far enough to perform an update

        (disp_lin, disp_ang) = pose_displacement(
            p1=pose_current,
            p2=self.state.pose_previous,
        )

        if (
            v2.magnitude(disp_lin) < ParticleFilterBase.lin_mvmt_threshold
            and disp_ang < ParticleFilterBase.ang_mvmt_threshold
        ):
            return

        particle_cloud = self.update_particles(
            particle_cloud=self.state.particle_cloud,
            disp_linear=disp_lin,
            disp_angular=disp_ang,
            laser_scan=data,
        )

        robot_estimate = self.update_estimated_robot_pose(
            particle_cloud,
        )

        self.state = Initialized(
            particle_cloud=particle_cloud,
            pose_previous=pose_current,
        )

        self.publish_particle_cloud(particle_cloud)
        self.publish_estimated_robot_pose(robot_estimate)

    def update_particles(
        self,
        particle_cloud: List[Particle],
        disp_linear: Vector2,
        disp_angular: float,
        laser_scan: LaserScan,
    ) -> List[Particle]:
        with_motion = self.update_particles_with_motion_model(
            particle_cloud,
            disp_linear,
            disp_angular,
        )

        with_measurement = self.update_particle_weights_with_measurement_model(
            particle_cloud=with_motion,
            laser_scan=laser_scan,
        )

        normalized = self.normalize_particles(
            particle_cloud=with_measurement,
        )

        resampled = self.resample_particles(
            particle_cloud=normalized,
        )

        return resampled

    @abstractmethod
    def initialize_particle_cloud(
        self,
        num_particles: int,
        occupancy_grid: OccupancyGrid,
    ) -> List[Particle]:
        pass

    @abstractmethod
    def normalize_particles(self, particle_cloud: List[Particle]) -> List[Particle]:
        # make all the particle weights sum to 1.0
        pass

    @abstractmethod
    def resample_particles(self, particle_cloud: List[Particle]) -> List[Particle]:
        pass

    @abstractmethod
    def update_estimated_robot_pose(self, particle_cloud: List[Particle]) -> TurtlePose:
        # based on the particles within the particle cloud, update the robot pose estimate
        pass

    @abstractmethod
    def update_particle_weights_with_measurement_model(
        self,
        particle_cloud: List[Particle],
        laser_scan: LaserScan,
    ) -> List[Particle]:
        pass

    @abstractmethod
    def update_particles_with_motion_model(
        self,
        particle_cloud: List[Particle],
        disp_linear: Vector2,
        disp_angular: float,
    ) -> List[Particle]:
        # based on the how the robot has moved (calculated from its odometry), we'll  move
        # all of the particles correspondingly
        pass
