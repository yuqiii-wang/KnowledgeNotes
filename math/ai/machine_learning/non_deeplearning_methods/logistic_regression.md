# Logistic Regression

## Generalized Linear Models (GLMs)

The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

## Motivation

Consider independent binary tests $y_i \in \{0,1\}$, by Bernoulli distribution to model the tests: $y_i \sim \text{Bernoulli}(\pi_{i})$, where $\pi_i$ is the probability of $y_i=1$.

$$
\begin{align*}
f(y_i; \pi_i) &= \pi^{y_i}(1-\pi_i)^{1-y_i}
\\ &=
e^{y_i \log\pi_i + (1-y_i) \log(1-\pi_i)}
\\ &=
e^{y_i \log(\frac{\pi_i}{1-\pi_i}) + \log(1-\pi_i)}
\end{align*}
$$

The exponential form of $f(y_i; \pi_i)$ is taken to conform Generalized Linear Models (GLMs), where the mean and variance can be computed easily.

Set $\theta_i=\log(\frac{\pi_i}{1-\pi_i})$ and $b(\theta_i)=\log(1-\pi_i)=\log\big(\exp(\theta_i)+1 \big)$, then

$$
\begin{align*}
f(y_i; \pi_i) &=
e^{y_i \log(\frac{\pi_i}{1-\pi_i}) + \log(1-\pi_i)}
\\ &=
e^{y_i\theta_i+b(\theta_i)}
\end{align*}
$$

The first order derivative $\mu_i = b'(\theta_i)$ is same as Bernoulli mean $\pi_i$.
$$
\mu_i = b'(\theta_i) = 
\frac{\exp(\theta_i)}{\exp(\theta_i)+1}
= \pi_i
$$

## Function Form

<div style="display: flex; justify-content: center;">
      <img src="imgs/lr.png" width="40%" height="40%" alt="lr" />
</div>
</br>

$$
p(x) = 
\frac{\exp(\theta_i)}{\exp(\theta_i)+1}
=
\frac{1}{1+e^{-\frac{x-\mu}{s}}}
$$