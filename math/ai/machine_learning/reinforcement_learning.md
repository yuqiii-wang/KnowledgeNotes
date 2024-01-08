# Reinforcement Learning

Reinforcement learning describes how to guide actions $a \in A$ of an agent given a set of environments and agent states $s \in S$ to achieve high/maximize reward objective function $R(s,s')$ from state $s$ to $s'$.
In this process, $P_a(s,s')=P(S_{t+1}=s' | S_t=s, A_t=a)$ is the probability distribution computed at the time $t$ from state $s$ to $s'$.

* Policy

The agent's action selection is modeled as a map called *policy* $\pi$ to a probability $0 \le \pi(a,s) \le 1$.

$$
\begin{align*}
\pi(a,s) = P(A_t = a | S_t = s)    
\end{align*}
$$

In other words, a policy is the action probability distribution conditioned on states.

For example, a group of people (aka agent) traveling on a land mass (described in this figure) to destination should have below actions and states.

<div style="display: flex; justify-content: center;">
      <img src="imgs/policy_travel_example.png" width="50%" height="30%" alt="policy_travel_example" />
</div>
</br>

|Actions|States|
|-|-|
|$a_1=\text{TravelByForest}$|$s_0=\text{OnStartTile}$|
|$a_2=\text{TravelByGrassland}$|$s_1=\text{OnGrassland}$|
|$a_3=\text{TravelByPlain}$|$s_2=\text{OnPlainTile}$|
||$s_3=\text{OnForestTile}$|
||$s_4=\text{OnDestinationTile}$|

The policy $\pi(a,s) = P(A_t = a | S_t = s)$ can be summarized to the below, that each column's values should sum up to $1.0$, except for $\text{OnDestinationTile}$ where no further action is required as already reached the destination.

Since grassland is easy to transpass compared to forest, this policy assigns higher probability to $\text{TravelByGrassland}$ than to $\text{TravelByForest}$.
In practice, policy $\pi(a,s)$ is learned via optimization.

$$
\begin{matrix}
                        & \text{OnStartTile} & \text{OnDestinationTile} & \text{OnPlainTile} & \text{OnForestTile} & \text{OnGrasslandTile} \\
\text{TravelByPlain}    & 1.0                & \text{Not Available}     & 0.0                & 0.0                 & 0.0                    \\
\text{TravelByGrassland}& 0.0                & \text{Not Available}     & 0.8                & 0.0                 & 1.0                    \\
\text{TravelByForest}   & 0.0                & \text{Not Available}     & 0.2                & 1.0                 & 0.0                    \\
\end{matrix}
$$


* Optimization Objective

Optimization objective $J$ is defined as *expected discounted return* to be maximized.

Set $G$ as the discounted return at a time $t=0$, and $R_{t+1}$ is the reward for transitioning from state $S_t$ to $S_{t+1}$.
Having included $0 \le \gamma < 1$ served as a discount rate that future rewards $R_t$, where $t > 0$, are discounted then summed to the total discounted return.

$$
G = \sum^{\infty}_{t=0} \gamma^t R_{t+1} =
R_t + \gamma R_2 + \gamma^2 R_3 + ... 
$$

$\gamma^t R_{t+1}$ says for distant future where $t \rightarrow +\infty$, the reward $\gamma^t R_{t+1}$ will be very small.
This renders rewards get weighted more on recent time rather than the faraway future.

Expected discounted return $J$ at a time $t=0$ given an init state $S_0=s$ is the objective to be maximized by choosing the optimal $\pi$.

$$
\max_{\pi} J =
\max_{\pi} \mathbb{E} \big( G | S_0 = s, \pi \big) =
\max_{\pi} \mathbb{E}\bigg( G = \sum^{\infty}_{t=0} \gamma^t R_{t+1} \space\bigg|\space S_0 = s, \pi \bigg)
$$

## Markov Decision Process

## Policy Gradient

Typically, to find the optimal parameter set $\theta^*$ from optimization, policy gradient approach is to maximize expectation such that $\theta^*=\argmax_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\big( r(\tau) \big)$, where $r(\tau)$ is a reward function (opposite to a loss function in machine learning, be maximized instead of being minimized).

Optimize the objective by gradient ascent:

$$
\theta_{k+1}= \theta_{k} + \eta \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\big( r(\tau) \big)
$$