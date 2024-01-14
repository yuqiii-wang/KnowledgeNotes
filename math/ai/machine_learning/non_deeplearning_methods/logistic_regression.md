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

Here, want to introduce a parameter group $\Theta=\{ \theta_i \}$ that finds the maximum likelihood of $f(y_i; \pi_i)$.

Set $\theta_i=\log(\frac{\pi_i}{1-\pi_i})$ and $b(\theta_i)=\log(1-\pi_i)=\log\big(\exp(\theta_i)+1 \big)$, then

$$
\begin{align*}
f(y_i; \pi_i) &=
e^{y_i \log(\frac{\pi_i}{1-\pi_i}) + \log(1-\pi_i)}
\\ &=
e^{y_i\theta_i+b(\theta_i)}
\\ \log f(y_i; \pi_i)&= y_i\theta_i+b(\theta_i)
\end{align*}
$$

The maximum log likelihood can be found by
$$
\begin{align*}
0 &= \frac{\partial \log f(\Theta=\theta)}{\partial \pi_i}
\\ 
0 &= \sum^n_{i=1} \Big(  \Big)
\end{align*}
$$

The $\theta_i=\log(\frac{\pi_i}{1-\pi_i})$ is used to map the probability to $(0, +\infty)$, where probability $\pi_i \approx 0.5$ sees drastic changes, while $\pi_i \approx 0.99$ and $\pi_i \approx 0.01$ see little changes in output values.

This is useful such as in activation function in deep learning that it outputs most/little of the "energy" when the result is almost certain to be true/false, rather than a simple linear "energy" output.

The first order derivative $\mu_i = b'(\theta_i)$ is same as Bernoulli mean $\pi_i$.
$$
\mu_i = b'(\theta_i) = 
\frac{\exp(\theta_i)}{\exp(\theta_i)+1}= \pi_i
$$

## Function Form

<div style="display: flex; justify-content: center;">
      <img src="imgs/lr.png" width="20%" height="20%" alt="lr" />
</div>
</br>

$$
p(x) = 
\frac{\exp(\theta_i)}{\exp(\theta_i)+1}=
\frac{1}{1+e^{-\frac{x-\mu}{s}}}
$$