# Policy Gradient

Typically, to find the optimal parameter set $\theta^*$ from optimization, policy gradient approach is to maximize expectation such that $\theta^*=\argmax_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\big( r(\tau) \big)$, where $r(\tau)$ is a reward function (opposite to a loss function in machine learning, be maximized instead of being minimized).

Optimize the objective by gradient ascent:

$$
\theta_{k+1}= \theta_{k} + \eta \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\big( r(\tau) \big)
$$