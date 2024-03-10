# Logistic Regression

## Motivation and Deduction

Consider independent binary tests $y_i \in \{0,1\}$, by Bernoulli distribution to model the tests: $y_i \sim \text{Bernoulli}(\pi_{i})$, where $\pi_i$ is the probability of $y_i=1$ (the opposite is $1-\pi_i$ for $y_i=0$).

A sequence of Bernoulli trials $y_1, y_2, ..., y_n$ with a constant probability $\pi_i$ is

$$
\prod^{n}_{i=1} f(y_i; \pi_i)^{y_i} \big(1-f(y_i; \pi_i)\big)^{1-y_i} =
\prod^{n}_{i=1} \pi_i^{y_i} (1-\pi_i)^{1-y_i}
$$

Usually, likelihood is maximized at $\hat{p}=\frac{1}{n} \sum^n_{i=1} y_i$.
However, if it is known that the result $y_i$ is dependent on conditions/inputs, e.g., $\bold{x}_i \rightarrow y_i$, the likelihood $\pi_i$ can be better approached by taking consideration the conditions/inputs $\bold{x}_i$.

Now, the problem is to map real inputs to a probability such that $\bold{x} \in \mathbb{R} \rightarrow y \in \mathbb{R} \rightarrow \pi \in [0,1]$, and $y=\beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$ is a typical implementation taking linear form.

$y=\ln\frac{\pi}{1-\pi}$ is proposed for $\pi \in [0,1]$, the mapping is to all numbers in real $y \in \mathbb{R}$.
And $\pi=\frac{e^y}{1+e^y}$ (the *sigmoid function*) is proposed so that $e^y$ can be easily computed/mapping $\bold{x} \rightarrow y$.

Illustrated as below, need to define $\pi$ so that $\frac{f(y=1; \pi)}{f(y=0; \pi)} > 0$ can be covered in all positive real numbers, and there is monotony that, when most of $y_i$ are $y_i=1$, there is $\lim\frac{f(y=1; \pi)}{f(y=0; \pi)}=+\infty$ and when most of $y_i$ are $y_i=0$, there is $\lim\frac{f(y=1; \pi)}{f(y=0; \pi)}=-\infty$.

$$
\begin{align*}
\frac{f(y=1; \pi)}{f(y=0; \pi)} &= \frac{\pi}{1-\pi} \\
&= \frac{\frac{e^y}{1+e^y}}{1-\frac{e^y}{1+e^y}} \\
&= \frac{e^y}{(e^y+1)\Big(\frac{1+e^y}{1+e^y}-\frac{e^y}{1+e^y}\Big)} \\
&= \frac{e^y}{(e^y+1)\Big(\frac{1+e^y-e^y}{1+e^y}\Big)} \\
&= \frac{e^y}{(e^y+1)\Big(\frac{1}{1+e^y}\Big)} \\
&= e^y \\
\Rightarrow \qquad \ln \frac{\pi}{1-\pi} &= y \\
\Rightarrow \qquad\qquad\quad y &= \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... \qquad \text{linear expression of }y
\end{align*}
$$

This is useful such as in activation function in deep learning that it outputs most/little of the "energy" when the result is almost certain to be true/false, rather than a simple linear "energy" output.

### Derivative

Given $\pi(y)=\frac{e^y}{1+e^y}$, the derivative is $\pi'(y)=\pi(y)\big(1-\pi(y)\big)$.

Interestingly, $\pi'(x)$ is exactly a step of a sequence of Bernoulli trials $\prod^{n}_{i=1} \pi_i^{y_i} (1-\pi_i)^{1-y_i}$ where $$

## Function Form

<div style="display: flex; justify-content: center;">
      <img src="imgs/lr.png" width="40%" height="30%" alt="lr" />
</div>
</br>

$$
y = 
\frac{\exp(x)}{\exp(x)+1}=
\frac{1}{1+e^{-\frac{x-\mu}{s}}}
$$
