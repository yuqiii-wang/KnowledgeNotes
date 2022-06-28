# Dynamic Window Approach to Collision Avoidance

On a 2-d motion space, $[x, y, \theta]$ describes the kenimatic configuration of a robot. Define $v(t)$ as translational velocity and $\omega(t)$ as angular velocity, there is
$$
x(t) = \int ^{t_n}_{t_0} v(t) \cdot cos\theta(t) \space dt
\\
y(t) = \int ^{t_n}_{t_0} v(t) \cdot sin\theta(t) \space dt
$$

Approximation discretize the continuous model with robot's velocities $[v(t), \omega(t)]$ being constant during an interval $[t_i, t_{i+1}]$. We use $[v_i, \omega_i]$ to represent velocities in discrete space.

## Search Space and Optimization

### Search Space

* Admmisible Velocities

$[v, \omega]$ is constraint by hardware acceleration, forbidden abrupt changes in $[t_k, t_{k+1}]$.

Given the max acceleration $[v_{max}', \omega_{max}']$, admissible velocities have another constraint that a robot should be able to stop before collision to its nearby obstacles.

* Trajectories

Max possible curvature trajectories permitted by admissible velocities $[v, \omega]$

### Objective Function

The objectuve function to be optimized is 
$$
\sigma
\big(
    \alpha \cdot heading(v,\omega) + \beta \cdot dist(v,\omega) + \gamma \cdot velocity(v, \omega)
\big)
$$
where we want to maximize
* heading: robot's movement toward goal
* distance/clearance: robot's distance to its nearby obstacles
* velocity: robot should be encouraged to move as fast as possible
