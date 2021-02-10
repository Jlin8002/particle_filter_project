# particle_filter_project

**Team: Jason Lin and Adam Weider**

### TurtleBot3 localizes itself with the help of the particle filter

![particle filter run](media/particle_filter.gif)

The above run was performed with 5000 particles, rather than the normal
10,000, in order to have the cloud follow TurtleBot more closely during the
demonstration.

## Setup

This project uses `pipenv` for local package management. [Read here](docs/setup.md)
for instructions on how to perform setup.

## Objectives

Our goal is to implement a particle filter to learn about the processes involved
in robot localization. We will then apply our particle filter to a scenario in
which a robot needs to ascertain its position in order to properly navigate out
of a room.

## High-level description

Our implementation uses a particle filter and likelihood field to compare a robot's
laser scan readings to a map of its surroundings. After initializing the particle
cloud, we update each of the particle poses and resample them as the robot moves
throughout the map. During a single update, we calculate the robot's translational
and rotational displacement from the previous update and apply the displacements
(with a small amount of noise) to each of the particles. We then check the robot's
laser scan measurements against a particle's surroundings and use a likelihood
field to calculate the probability of a given particle being in the same position
as the robot. Once all the particles have been updated, we resample our particles
based on those probabilities to converge on a location near the robot's actual
position.

## Main steps

We implemented operations for updating the particle cloud in
[`scripts/particle_cloud.py`](scripts/particle_cloud.py). The functions performing
the main operations are:

- `initialize` for creating the initial particle cloud from the environment map.
- `update_poses_and_weights` for handling robot movement (motion model) and laser
scans (measurement model).
  - We merged these operations into one to save a pass over the particle cloud.
- `normalize` for normalizing particle weights after the measurement update.
- `resample` for resampling the particles after weight normalization.

We included docstrings and comments for documenting the above operations (and
the rest of the codebase).

Although not a requirement of the project, we wrote our own
[small framework](scripts/particle_filter.py) for running the particle cloud,
using ROS helper modules originally written during the warmup project
([repository here](https://github.com/AHW214/rospy-util)).

## Challenges

We faced challenges when figuring out how to deal with laser scan distances greater
than 3.5 meters (the max scan distance) and likelihood field lookups falling outside
of the map. We knew that both scenarios would involve either skipping a measurement
or applying a penalty to the weights, but we weren't sure what form that would
take until we tried implementing them. Luckily, we quickly ascertained that skipping
measurements greater than 3.5 and applying a penalty to unknown likelihood field
values would give us a particle filter with decent convergence behavior.

We also ran into an issue where particles would incorrectly converge to positions
in similar-looking rooms due to gaps in our scan angles. Depending on how many
angles we chose to incorporate into our measurement model, the particles would
always converge in some locations but not in others. We fixed this by making two
changes: a correction to our measurement model so that it only sampled scan
measurements from ranges under the maximum scan distance, and tweaks to our
noise/Gaussian width parameters to slow convergence (thus sustaining particles
near the robot, but not yet exactly at the robot).

## Future work

If we had more time, we would implement a beam measurement model to compare its
performance to our likelihood field model. We recognize that the beam model may
improve the behavior of particles near the walls of the map, whose simulated scans
may fall out of bounds. Scaling particle weights by the distance out of bounds,
instead of applying a flat penalty as we currently do, will preserve particles
that are near the robot, but happen to be against the boundaries of the map.

## Takeaways

- When tuning our particle filter parameters, our initial changes rendered our
localization ineffective. Our impression was thus that our filter was tuned, and
that we should not adjust it further. We then considered that, although the changes
seemed small, perhaps an even finer tuning was possible. After halving the
adjustment (an addition of 0.05 meters to our obstacle distance SD, which seemed
inconsequential), the filter localized even more effectively than before making
changes. The takeaway would be that, for making adjustments to the parameters of
the system, have a sense of scale for each parameter. That way you do not tune
too heavily or lightly, and assume the system is already tuned (as we almost did).

- We learned the importance of building preventative systems into our algorithms
instead of turning to reactionary measures in the event that unexpected behavior
occurred. When we discovered that our particle cloud did not localize properly
at times, we first devised a way to scatter particles based on their weights in
the hopes that one of them would land near the robot's true position and make
the cloud converge; however, we resolved the issue in a much less convoluted
fashion by realizing that we could slow the particle cloud's convergence, giving
the filter more time to evaluate whether it happened to be in the wrong spot. This
saved us time and effort, ultimately improving our algorithm's performance.

- Although git makes asynchronous group work relatively painless (except when
resolving merge conflicts), we found it really helpful calling and working
synchronously. When programming, we verified each other's edits, offering a safety
net of sorts. When debugging, we shared our perspectives on the issues at hand,
combining our logic to more quickly identify and address those problems. Finally,
working together helped us maintain presence and focus for the duration of the
project.

- Using the Visual Studio Code [Live Share](https://code.visualstudio.com/learn/collaboration/live-share)
extension greatly increased the efficacy of pair programming remotely. Being able
to edit the same code in real time meant that we could apply changes across our
project a lot more quickly and cut down on time spent fixing merge conflicts. Think
Google Docs but for code.
