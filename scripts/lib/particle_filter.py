import math
from typing import List, Optional, Tuple

import rospy

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from tf import TransformBroadcaster, TransformListener

from lib.particle import Particle
from lib.util import points_distance, yaw_from_pose


def pose_displacement(p1: PoseStamped, p2: PoseStamped) -> Tuple[float, float]:
    displacement_linear = points_distance(p1.pose.position, p2.pose.position)
    displacement_angular = abs(yaw_from_pose(p2.pose) - yaw_from_pose(p1.pose))

    return (displacement_linear, displacement_angular)


class ParticleFilter:
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

        # once everything is setup initialized will be set to true
        self.initialized: bool = False

        # initialize this particle filter node
        rospy.init_node("turtlebot3_particle_filter")

        self.map: Optional[OccupancyGrid] = None

        # initialize the particle cloud array
        self.particle_cloud: List[Particle] = []

        # initialize the estimated robot pose
        self.robot_estimate: Pose = Pose()  # TODO optional

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
        rospy.Subscriber(ParticleFilter.map_topic, OccupancyGrid, self.get_map)

        # subscribe to the lidar scan from the robot
        rospy.Subscriber(ParticleFilter.scan_topic, LaserScan, self.robot_scan_received)

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

        header = Header(stamp=rospy.Time.now(), frame_id=ParticleFilter.map_topic)

        poses = map(lambda p: p.pose, self.particle_cloud)

        particle_cloud_pose_array = PoseArray(header, poses)

        self.particles_pub.publish(particle_cloud_pose_array)

    def publish_estimated_robot_pose(self):

        header = Header(stamp=rospy.Time.now(), frame_id=ParticleFilter.map_topic)

        pose = self.robot_estimate

        robot_pose_estimate_stamped = PoseStamped(header, pose)

        self.robot_estimate_pub.publish(robot_pose_estimate_stamped)

    def resample_particles(self) -> None:
        # TODO

        pass

    def robot_scan_received(self, data: LaserScan) -> None:

        # wait until initialization is complete
        if not self.initialized:
            return

        # we need to be able to transfrom the laser frame to the base frame
        if not (
            self.tf_listener.canTransform(
                ParticleFilter.base_frame, data.header.frame_id, data.header.stamp
            )
        ):
            return

        # wait for a little bit for the transform to become avaliable (in case the scan arrives
        # a little bit before the odom to base_footprint transform was updated)
        self.tf_listener.waitForTransform(
            ParticleFilter.base_frame,
            ParticleFilter.odom_frame,
            data.header.stamp,
            rospy.Duration(0.5),
        )

        if not (
            self.tf_listener.canTransform(
                ParticleFilter.base_frame, data.header.frame_id, data.header.stamp
            )
        ):
            return

        # calculate the pose of the laser distance sensor
        laser_pose = PoseStamped(
            header=Header(stamp=rospy.Time(0), frame_id=data.header.frame_id)
        )

        self.laser_pose = self.tf_listener.transformPose(
            ParticleFilter.base_frame, laser_pose
        )

        # determine where the robot thinks it is based on its odometry
        odom_pose = PoseStamped(
            header=Header(stamp=data.header.stamp, frame_id=ParticleFilter.base_frame),
            pose=Pose(),
        )

        self.odom_pose = self.tf_listener.transformPose(
            ParticleFilter.odom_frame, odom_pose
        )

        # we need to be able to compare the current odom pose to the prior odom pose
        # if there isn't a prior odom pose, set the odom_pose variable to the current pose
        if self.odom_pose_last_motion_update is None:
            self.odom_pose_last_motion_update = self.odom_pose
            return

        if self.particle_cloud:

            # check to see if we've moved far enough to perform an update

            (disp_lin, disp_ang) = pose_displacement(
                p1=self.odom_pose_last_motion_update,
                p2=self.odom_pose,
            )

            if (
                disp_lin > ParticleFilter.lin_mvmt_threshold
                or disp_ang > ParticleFilter.ang_mvmt_threshold
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

    def update_particle_weights_with_measurement_model(self, data: LaserScan) -> None:
        # TODO

        pass

    def update_particles_with_motion_model(self):
        # based on the how the robot has moved (calculated from its odometry), we'll  move
        # all of the particles correspondingly

        # TODO

        pass
