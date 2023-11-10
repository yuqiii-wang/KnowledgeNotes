# Optimizers

## Stochastic Gradient Descent with Momentum (SGDM)

SGD with Momentum is a stochastic optimization method that adds a momentum term.

The direction of the previous update is retained to a certain extent during the update, retaining inertia of movement.

Update $\Delta W$ at the $n$-th iteration is defined as
$$
\Delta W_{n+1} = \alpha \Delta W_{n} + \eta \frac{\partial\space Loss}{\partial\space W_{n}}
$$
where $\alpha$ is the momentum rate and $\eta$ is the learning rate.

## Adagrad (Adaptive Gradient Descent)

Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the updates.

Update $\Delta W$ at the $n$-th iteration is defined as
$$
\Delta W_{n+1} = 
\frac{\eta}{\sqrt{cache_{n+1}}+\epsilon}
\cdot
\frac{\partial\space Loss}{\partial\space W_{n}}
$$
where $\eta$ is the learning rate and $\epsilon$ is a very small constant preventing division by zero error when $cache$ is nearly or equal to zero.

$cache$ is computed as
$$
cache_{n+1} = 
cache_n + \big(\frac{\partial\space Loss}{\partial\space W_{n}}
\big)^2
$$

## RMS-Prop (Root Mean Square Propagation)

RMS-Prop is a special version of Adagrad in which the learning rate is a moving average of the recent two gradient instead of the cumulative sum of squared gradients. 

$$
cache_{n+1} = 
\gamma \cdot cache_n + (1-\gamma)\cdot\big(\frac{\partial\space Loss}{\partial\space W_{n}}
\big)^2
$$
where $\gamma=0.9$ is a typical setting, and the weight update is the same as that of Adagrad.
$$
\Delta W_{n+1} = 
\frac{\eta}{\sqrt{cache_{n+1}}+\epsilon}
\cdot
\frac{\partial\space Loss}{\partial\space W_{n}}
$$

## Adam (Adaptive Moment Estimation)

Adam makes use of the average of the first and second moments of the gradients. 

The parameters of $\beta_1$ and $\beta_2$ are used to control the decay rates of these moving averages. 

The first and second order momentum at the $n$-th iteration:
$$
\begin{align*}
m_{n+1} &= \beta_1 m_{n} + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n}}
\\
v_{n+1} &= \beta_2 v_{n} + (1-\beta_2)\big( \frac{\partial\space Loss}{\partial\space W_{n}} \big)^2
\end{align*}
$$

Define the bias-corrected momentums $\hat{m}_{1,n+1}$ and $\hat{m}_{2,n+1}$
$$
\begin{align*}
\hat{m}_{n+1} &= \frac{m_{n+1}}{1-\beta_1^n}
\\
\hat{v}_{n+1} &= \frac{v_{n+1}}{1-\beta_2^n}
\end{align*}
$$

Finally, the weight update at the $n$-th iteration is
$$
\Delta W_{n+1} = 
\eta \frac{\hat{m}_{n+1}}{\sqrt{\hat{v}_{n+1}}+\epsilon}
$$

### Adam Derivation

Adam follows the philosophy that momentums at the $n$-th iteration should see their expected values be equal to the expected values over all $n$ history gradients. 

This thinking (*unbiased estimators*) can be expressed as below.
$$
\begin{align*}
E[m_n] &= E[g_n]
\\
E[v_n] &= E[g_n^2]
\end{align*}
$$ 

Given this consideration, $m_n$ and $v_n$ can be said good momentums with well captured recent gradients with high importance and remote gradients with low importance. The relative importance is materialized by decaying exponential $\beta_1^n, \beta_2^n$ for $0<\beta_1<1$ and $0<\beta_2<1$. 

Define gradients at the $n$-th iteration as $g_n = \frac{\partial\space Loss}{\partial\space W_{n}}$ and $g_n^2 = (\frac{\partial\space Loss}{\partial\space W_{n}})^2$.

Expanding the momentums, for example $m_n$, for $n=3$, there is
$$
\begin{align*}
m_3 &= \beta_1 m_2 + (1-\beta_1)g_3
\\ &= 
\beta_1 m_1 + \beta_1(1-\beta_1)g_2 + (1-\beta_1)g_3
\\ &=
\beta_1^2 (1-\beta_1)g_1 + \beta_1 (1-\beta_1)g_2 + (1-\beta_1)g_3
\end{align*}
$$

To summarize, there are ($v_n$ has similar derivation as well)
$$
\begin{align*}
m_n &= (1-\beta_1) \sum_{i=0}^n
\beta_1^{n-i} g_i
\\
v_n &= (1-\beta_2) \sum_{i=0}^n
\beta_2^{n-i} g_i^2
\end{align*}
$$

Consider the discrepancy between $E[m_n]$ and $E[g_n]$ ($v_n$ has similar deductions as well)
$$
\begin{align*}
E[m_n] &= 
E\bigg[
    (1-\beta_1) \sum_{i=1}^n \beta_1^{n-i} g_i
\bigg]
\\ &=
E[g_n] (1-\beta_1)\sum_{i=1}^n \beta_1^{n-i} + \xi
\quad\quad\quad &(1)
\\ &=
E[g_n] (1-\beta_1)  \frac{1-\beta_1^n}{1-\beta_1} + \xi
\quad\quad\quad &(2)
\\ &=
E[g_n] (1-\beta_1^n) + \xi
\end{align*}
$$
where $(1)$ is derived by taking out $E[g_n]$ from the summation and the remaining term is $\xi$, which represents the difference between sum of $\beta_1^{n-i} g_i$ and that of just applying $E[g_n]\beta_1^{n-i}$. $(2)$ is just the result of finite geometric sequence sum.

As a result, bias correction is done by cancelling the term $1-\beta_1^n$ with $\hat{m}_{n} = \frac{m_{n}}{1-\beta_1^n}$.

$v_n$ takes similar deduction result and has the correction term $\hat{v}_{n} = \frac{v_{n}}{1-\beta_2^n}$

### Loss Function Scaling Impact on ADAM

For SGD, by multiplying $0.5$ to loss function, the learning rate is reduced to half.
However, for ADAM, this is not true.

For $\frac{\partial\space Loss}{\partial\space W_{n}} \rightarrow m_{n+1}$ and $(\frac{\partial\space Loss}{\partial\space W_{n}})^2 \rightarrow v_{n+1}$, and finally for $\Delta W_{n+1} = \eta \frac{\hat{m}_{n+1}}{\sqrt{\hat{v}_{n+1}}+\epsilon}$, the square root operation cancels out the $0.5$ effect on the loss function.
In conclusion, scaling on loss function has no effect on ADAM learning/weight update.

## Bayesian Optimizer

