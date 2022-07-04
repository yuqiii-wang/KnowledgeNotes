# Gradient Descent

Gradient descent (also often called steepest descent) is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. 

Define a multi-variable function $\bold{F}(\bold{x})$ being differentiable in a neighbourhood of a point $a$, then $\bold{F}(\bold{x})$ decreases fastest updating $a_n$ to its local minimum following
$$
a_{n+1}=a_n+\gamma \Delta \bold{F}(\bold{x})
$$
where $\gamma$ is learning rate.