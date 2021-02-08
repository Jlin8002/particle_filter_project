# particle_filter_project

**Team: Jason Lin and Adam Weider**

## Outline of components

### Initializing the particle cloud

- Implementation: Take `ParticleFilter.map.data`, filter out black (boundary) and gray (out of bounds) values, leaving only white (navigable space) values. Then, randomly select `ParticleFilter.num_particles` of these values with replacement. Create poses for the particles, using the indices of these values for their x and y positional coordinates, and random [0, 2π) z-axis rotations for their orientations. Use initial weights of 1.0.

- Testing: Copy the original `OccupancyGrid` and replace the white values with new values, whose black level increases with the number of particles at the corresponding position (white if 0 particles, light gray if 1 particle, etc). Use `map_saver` to visualize the grid as a `.pgm` file.

### Updating the particles based on robot movement

- Implementation: Subtract the positions and rotations of `ParticleFilter.odom_pose_last_motion_update` from those of `ParticleFilter.odom_pose` to obtain the linear and angular displacement of the robot. Apply these displacements to each of the particles. To check updated positions, perform a lookup in the `OccupancyGrid`: if the updated location corresponds to a black or gray value, then the particle has moved outside the navigable space. Possible responses include: 1) Lower (or zero) the weight of the particle; 2) Use the old position; 3) Find and apply the maximum possible displacement to the particle.

- Testing: Use the `matplotlib` package to visualize the particles before and after a motion update.

### Computing importance weights

- Implementation: Perform a raycast for each particle to determine the distances, in cells, to the nearest obstacles (black values) up to the search radius of the scanner (search radius of LiDAR divided by the cell resolution). [Details discussed here](https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/) (possibly convoluted to implement, we might be over thinking this). Convert LiDAR scan distances to cells, and use the same weighting procedure from the class example (sum of differences).

- Testing: Test raycast operation with a single particle in a simple environment (no obstacles, single obstacle, one obstacle occluding another, etc) and verify the simulated scans. Then, test with many particles, and ensure that the highest weighted particles have a similar view to that of the robot.

### Normalizing importance weights

- Implementation: Sum the unnormalized weights and compute the reciprocal to find the normalization factor. Multiply all particle weights by this factor.

- Testing: Ensure there are no typos (we don't anticipate anything going seriously wrong).

### Resampling particles

- Implementation: Call `draw_random_sample` with the particle cloud as the elements from which to draw, the particle weights as the corresponding probabilities, and the number of particles as the number of samples to draw. Set the resulting sample as the particle cloud for the next iteration.

- Testing: Assuming `draw_random_sample` operates correctly, no testing should be necessary.

### Updating estimated robot pose

- Implementation: Compute the centroid of the particles to estimate position, and compute a weighted mean of their orientations to estimate rotation.

- Testing: Visually verify that the estimated pose converges on that of the robot (as the particles converge).

### Accounting for noise

- Implementation: Add noise to the distance measurements in the simulated particle scans.

- Testing: Observe how the simulated scan noise affects the performance of the filter. In response, tune the addition of noise to minimize time to convergence.

## Timeline of milestones

### Wednesday, February 3rd
- Jason: Normalization, resampling, pose estimation, noise
- Adam: Initialization, updating movement
- Together: Simulating scan and computing weights

### Monday, February 8th
- Tested, presentable for class studio time

### Wednesday, February 10th
- Write-up and videos finished

## Objectives
Our goal is to implement a particle filter to learn about the processes involved in robot localization. We will then apply our particle filter to a scenario in which a robot needs to ascertain its position in order to properly navigate out of a room.

## High-level description
Our implementation uses a particle filter and likelihood field to compare a robot's laser scan readings to a map of its surroundings. After initializing the particle cloud, we update each of the particle poses and resample them as the robot moves throughout the map. During a single update, we calculate the robot's translational and rotational displacement from the previous update and apply the displacements (with a small amount of noise) to each of the particles. We then check the robot's laser scan measurements against a particle's surroundings and use a likelihood field to calculate the probability of a given particle being in the same position as the robot. Once all the particles have been updated, we resample our particles based on those probabilities to converge on a location near the robot's actual position.

## Main steps
- Movement: particle_cloud.py (update_poses)
- Weight computation: particle_cloud.py (update_weights)
- Resampling: particle_cloud.py (resample)

## Challenges
We faced challenges when figuring out how to deal with laser scan values greater than 3.5 and likelihood field lookups falling outside of the map. We knew that both scenarios would involve either skipping a measurement or applying a penalty to the weights, but we weren't sure what form that would take until we tried implementing them. Luckily, we quickly ascertained that skipping measurements greater than 3.5 and applying a penalty to unknown likelihood field values would give us a particle filter with decent convergence behavior.  

## Future work
If we had more time, we would implement a beam measurement model to compare its performance to our likelihood field model. We recognize that the beam model may improve the behavior of particles near the walls of the map: Scaling particle weights by the scan distance error instead of applying a flat penalty as we currently do will preserve particles that fall between the robot and a wall.

## Takeaways
- 
- 
