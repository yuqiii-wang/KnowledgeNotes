# Poisson Distribution

Poisson distribution is, given average rate, to calculate the probability of number of a certain event that occurs in a certain period (assumed events are independent).

For example, given a hospital birth rate 3 babies per hour, poisson distribution describes the possibility of, e.g., next hour having 5 babies, or next two hours have 6 babies.

The definition is given as

$$
f(k;\lambda) =
\text{Pr}(X=k) =
\frac{\lambda^k e^{-\lambda}}{k!}
$$

where

* $k$ is the number of occurrences
* $\lambda=\text{E}(X)=\text{Var}(X)$ is average number of events, where $\text{E}(X)$ is the expectation and $\text{Var}(X)$ is the variance; given average rate $r$ and interval $t$, there is $\lambda=rt$.

## Derivation

Reference: https://www.pp.rhul.ac.uk/~cowan/stat/notes/PoissonNote.pdf

Consider the time interval $t$ broken into small intervals of length $\delta t$.
With applied $\lambda=rt$, there are probabilities $P(0; \delta t)$ for no event occurred and $P(1; \delta t)$ for event occurred.

$$
\begin{align*}
P(0; \delta t) &= 1 - \lambda \delta t \\\\
P(1; \delta t) &= \lambda \delta t
\end{align*}
$$

Now consider the no event scenario for the next $t$ period, there is $P(0; t + \delta t) = P(0;t)(1-\lambda \delta t)$, that can be written as

$$
\begin{align*}
    && P(0; t + \delta t) &=&& P(0;t)(1-\lambda \delta t) & \\\\
    \Rightarrow\qquad && \frac{P(0; t + \delta t)-P(0;t)}{\delta t} &= &&\lambda P(0;t) \\\\
    \text{take limit} \Rightarrow \qquad && \lim_{\delta t \rightarrow 0} \frac{P(0; t + \delta t)-P(0;t)}{\delta t} &= &&\lambda P(0;t) \\\\
    \Rightarrow \qquad && \frac{d P(0; t)}{d t} &=&& \lambda P(0;t) \\\\
    \text{integrating to find solution} \Rightarrow \qquad && P(0; t) = C e^{-\lambda t}
\end{align*}
$$

To find the constant $C$, try compute with a point, say $t=0$; then $P(0, 0)=1$ is obvious as there must be no event for no time internal $t=0$.
So that $1 = C e^{-\lambda \cdot 0}$ gives $C=1$.

Finally, there is

$$
P(0; t) = e^{-\lambda t}
$$

Similarly, for $k$-event $t$-interval scenario, here defines

$$
\begin{align*}
    P(k; t + \delta t)
    &= P(k;t)-P(k;t)\lambda \delta t + P(k-1;t)\lambda \delta t
\end{align*}
$$

where, on average the during $t + \delta t$ there should be $k$ events; the above equation separate the term into

1) all $k$ events occurred in the $t$ period
2) no event in the final $\delta t$
3) $k-1$ events occurred in the $t$ period plus one event in $\delta t$

$$
\begin{align*}
    && P(k; t + \delta t)
    &=&& P(k;t)-P(k;t)\lambda \delta t + P(k-1;t)\lambda \delta t \\\\
    \Rightarrow && \lim_{\delta t \rightarrow 0} \big(P(k; t+\delta t)-P(k; t)\big) &=&& \lim_{\delta t \rightarrow 0} \lambda \delta t \big(P(k-1;t)- P(k;t) \big) \\\\
    \Rightarrow &&\frac{d P(k; t)}{dt} &=&& \lambda \big(P(k-1;t)- P(k;t) \big)
\end{align*}
$$

That gives $P(1;t)=\lambda t e^{-\lambda t}$ and further generalization $P(k;t)=\frac{(\lambda t)^k e^{-\lambda t}}{k!}$.
