# Outline of components

### Initializing the particle cloud

- Implementation: Take `ParticleFilter.map.data`, filter out black (boundary) and
gray (out of bounds) values, leaving only white (navigable space) values. Then,
randomly select `ParticleFilter.num_particles` of these values with replacement.
Create poses for the particles, using the indices of these values for their x and
y positional coordinates, and random [0, 2π) z-axis rotations for their orientations.
Use initial weights of 1.0.

- Testing: Copy the original `OccupancyGrid` and replace the white values with
new values, whose black level increases with the number of particles at the
corresponding position (white if 0 particles, light gray if 1 particle, etc).
Use `map_saver` to visualize the grid as a `.pgm` file.

### Updating the particles based on robot movement

- Implementation: Subtract the positions and rotations of
`ParticleFilter.odom_pose_last_motion_update` from those of `ParticleFilter.odom_pose`
to obtain the linear and angular displacement of the robot. Apply these displacements
to each of the particles. To check updated positions, perform a lookup in the
`OccupancyGrid`: if the updated location corresponds to a black or gray value,
then the particle has moved outside the navigable space. Possible responses
include: 1) Lower (or zero) the weight of the particle; 2) Use the old position; 3)
Find and apply the maximum possible displacement to the particle.

- Testing: Use RViz to visualize the particles before and after a motion update.

### Computing importance weights

- Implementation: ~~Perform a raycast for each particle to determine the distances,
in cells, to the nearest obstacles (black values) up to the search radius of the
scanner (search radius of LiDAR divided by the cell resolution).
[Details discussed here](https://theshoemaker.de/2016/02/ray-casting-in-2d-grids/)
(possibly convoluted to implement, we might be over thinking this). Convert LiDAR
scan distances to cells, and use the same weighting procedure from the class example
(sum of differences).~~
Create a likelihood field from the occupancy grid using nearest-neighbor search.
Take certain angles around each of the particles and look at the corresponding
LiDAR scan distances, then perform lookups in the likelihood field at those
locations to compare the robot scan distances to the surroundings of each of the
particles. Scale the particle weights with how closely the scan distances match
each of the particle's likelihood values.

- Testing: Test ~~raycast operation~~ likelihood weighting with a single particle
in a simple environment (no obstacles, single obstacle, one obstacle occluding
another, etc) and verify the simulated scans. Then, test with many particles,
and ensure that the highest weighted particles have a similar view to that of the
robot.

### Normalizing importance weights

- Implementation: Sum the unnormalized weights and compute the reciprocal to find
the normalization factor. Multiply all particle weights by this factor.

- Testing: Ensure there are no typos (we don't anticipate anything going seriously
wrong).

### Resampling particles

- Implementation: Call `draw_random_sample` with the particle cloud as the elements
from which to draw, the particle weights as the corresponding probabilities, and
the number of particles as the number of samples to draw. Set the resulting sample
as the particle cloud for the next iteration.

- Testing: Assuming `draw_random_sample` operates correctly, no testing should be
necessary.

### Updating estimated robot pose

- Implementation: Compute the centroid of the particles to estimate position, and
compute a weighted mean of their orientations to estimate rotation.

- Testing: Visually verify that the estimated pose converges on that of the robot
(as the particles converge).

### Accounting for noise

- Implementation: Add noise to the distance measurements in the simulated particle
scans.

- Testing: Observe how the simulated scan noise affects the performance of the
filter. In response, tune the addition of noise to minimize time to convergence.

## Timeline of milestones

### Wednesday, February 3rd
- Jason: Normalization, resampling, pose estimation, noise
- Adam: Initialization, updating movement
- Together: Simulating scan and computing weights

### Monday, February 8th
- Tested, presentable for class studio time

### Wednesday, February 10th
- Write-up and videos finished
