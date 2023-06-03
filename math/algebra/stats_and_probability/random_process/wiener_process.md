# Wiener Process

Wiener process is a real-valued continuous-time stochastic process, where deviation/variance grows as time increases $W_t\sim N(0,t)$

The name is coined for investigations by American mathematician Norbert Wiener on the mathematical properties of the one-dimensional Brownian motion.
It is often also called Brownian motion due to its historical connection with the physical process of the same name originally observed by Scottish botanist Robert Brown.

It has the below properties:
* $W_0$ = 0 as start
* $W$ increment is independent: for $t>0$, the future increments $W_{t+\Delta t}-W_t$ over the time interval $\Delta t \ge 0$ are independent of the past value $W_s, s < t$
* $W$ has Gaussian increments: $W_{t+\Delta t} - W_t \sim N(0, \Delta t)$
* $W$ is continuous on $t$

Given the above properties, Wiener process can be summarized to 
$$
W_t = W_t - W_0 \sim N(0,t)
$$

## Covariance and Error Propagation

$$
\begin{align*}
\text{cov}(W_{t1}, W_{t2}) &= 
E\Big( \big(W_{t1}-E(W_{t1})\big) \cdot \big(W_{t2}-E(W_{t2}) \big) \Big)
\\ &=
E(W_{t1} \cdot W_{t2})
&& \qquad \text{for } E(W_{t1})=0 \text{ and } E(W_{t2})=0
\\ &=
E \Big(W_{t1} \cdot \big( (W_{t2} - W_{t1}) + W_{t1} \big) \Big)
&& \qquad \text{substitute with } W_{t2} = (W_{t2} - W_{t1}) + W_{t1}
\\ &=
E \Big( 
\underbrace{W_{t1} \cdot \big( W_{t2} - W_{t1} \big)}_{
    = E(W_{t1}) \cdot  E(W_{t2}-W_{t1}) = 0 }
\Big) + E \big( W_{t1}^2 \big)
&& \qquad \text{for } W_{t1} \text{ and } W_{t2}-W_{t1} \text{ are independent}
\\ &= 
E \big( W_{t1}^2 \big)
\\ &=
t_1
\end{align*}
$$

Error propagation can be formulated/simulated as below, where $Z$ is an independent standard normal variable, representing one time test noise from the interval $t_2$ to $t_1$.
$$
W_{t2} = W_{t1} + \sqrt{t_2 - t_1} \cdot Z
,\qquad Z \sim N(0, t_2 - t_1)
$$