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

```python
## A simplt forward for classification
hx = np.sigmoid(np.dot(Wxh, x) + bx)
hh = np.sigmoid(np.dot(Whh, hx) + bh)
y_pred = np.softmax(np.dot(Why, hh) + by)
loss = np.dot(t_truth, y_pred) # cross-entropy loss

## A simple back-prop
dy = y_pred - 1 # back-prop into y by
dby += dy
...

## perform parameter update with Adagrad
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) # memory variables for Adagrad
mbx, mbh, mby = np.zeros_like(bx), np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
for param, dparam, cache in zip([Wxh, Whh, Why, bx, bh, by], 
                            [dWxh, dWhh, dWhy, dbx, dbh, dby], 
                            [mWxh, mWhh, mWhy, mbx, mbh, mby]):
cache += dparam ** 2
param += -learning_rate * dparam / np.sqrt(cache + 1e-8)
```


## RMS-Prop (RMSP, Root Mean Square Propagation)

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

The parameters of $\beta_1 \ge 0.9$ and $\beta_2 \ge 0.99$ are used to control the decay rates of these moving averages.

The first and second order momentum at the $n$-th iteration:

$$
\begin{align*}
m_{n+1} &= \beta_1 m_{n} + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n}}
\\
v_{n+1} &= \beta_2 v_{n} + (1-\beta_2)\big( \frac{\partial\space Loss}{\partial\space W_{n}} \big)^2
\end{align*}
$$

Define the bias-corrected momentums $\hat{m}\_{1,n+1}$ and $\hat{m}\_{2,n+1}$

$$
\begin{align*}
\hat{m}\_{n+1} &= \frac{m_{n+1}}{1-\beta_1^n}
\\
\hat{v}\_{n+1} &= \frac{v_{n+1}}{1-\beta_2^n}
\end{align*}
$$

Finally, the weight update at the $n$-th iteration is

$$
\Delta W_{n+1} = 
\eta \frac{\hat{m}\_{n+1}}{\sqrt{\hat{v}\_{n+1}}+\epsilon}
$$

### Adam Derivation

Adam follows the philosophy that momentums at the $n$-th iteration should see their expected values be equal to the expected values over all $n$ history gradients ($g_n = \frac{\partial\space Loss}{\partial\space W_{n}}$ and $g_n^2=(\frac{\partial\space Loss}{\partial\space W_{n}})^2$).

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

where $(1)$ is derived by taking out $E[g_n]$ from the summation and the remaining term is $\xi$, which represents the difference between sum of $\beta_1^{n-i} g_i$ and that of just applying $E[g_n]\beta_1^{n-i}$. $(2)$ is just the result of finite geometric sequence sum $\sum^n\_{i=1}r^i = \frac{a_1(1-r^n)}{1-r}$ for $r \ne 1$.

As a result, bias correction is done by cancelling the term $1-\beta_1^n$ with $\hat{m}\_{n} = \frac{m_{n}}{1-\beta_1^n}$.

$v_n$ takes similar deduction result and has the correction term $\hat{v}\_{n} = \frac{v_{n}}{1-\beta_2^n}$.

### Adam Variants: Weight Decay and AdamW

* Adam With Weight Decay

Before computing ADAM, add a small amount of previous iteration parameters $\lambda W_{n-1}$ to gradient, where $\lambda=0.01$.

$$
g_n \leftarrow g_{n} + \lambda W_{n-1}
$$

* AdamW

Before computing ADAM, add a small amount of previous iteration parameters $\lambda \eta W_{n-1}$ to this iteration parameters, where $\lambda=0.01$ for a learning rate $\eta=10^{-4}$.

$$
W_{n} \leftarrow W_{n-1} + \lambda \eta W_{n-1}
$$

### Choices of $\beta_1$ and $\beta_2$

$\beta_1 \ge 0.9$ and $\beta_2 \ge 0.99$ are typical choices.

As iterations grow $n \rightarrow +\infty$, there is

$$
\begin{align*}
    \lim_{n \rightarrow +\infty} \hat{m}\_{n} &=
    \frac{m_{n}}{1-\beta_1^{n-1}}
\\ &\approx
    m_{n} 
\\ &=
    \beta_1 \Big( \beta_1 m_{n-1} + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n-1}} \Big) + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n}}
\\ &=
    \beta_1 \Big( \beta_1 \Big(\beta_1 m_{n-2} + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n-2}} \Big) + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n-1}} \Big) + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n}}
\\ &=
    ...
\\ &=
    \beta_1^n m_0 +
    (1-\beta_1)\Big( \beta_1^n \frac{\partial\space Loss}{\partial\space W_{1}} + ... + \beta_1^2 \frac{\partial\space Loss}{\partial\space W_{n-2}}
    + \beta_1 \frac{\partial\space Loss}{\partial\space W_{n-1}} +
    \frac{\partial\space Loss}{\partial\space W_{n}} \Big)
\end{align*}
$$

By approximating $\beta_1^n \approx \beta_1^{n-1} \approx ... \approx 0$ for $n \gg 0$, when $\beta_1$ is small, the momentum $m_n$ is almost identical to $\frac{\partial\space Loss}{\partial\space W_{n}}$; when $\beta_1$ is large, the recent momentum info is retained, 
such as $ \beta_1^2 \frac{\partial\space Loss}{\partial\space W_{n-2}} + \beta_1 \frac{\partial\space Loss}{\partial\space W_{n-1}} + \frac{\partial\space Loss}{\partial\space W_{n}}$ has sufficient gradient info.

In other words, a large $\beta_1 \ge 0.9$ retains rich info about momentum.

Similarly for $v_n$, there is

$$
v_n =
\beta_2^n v_0 +
    (1-\beta_2)\Big( \beta_2^n \big( \frac{\partial\space Loss}{\partial\space W_{1}}\big)^2 + ... + \beta_2^2 \big( \frac{\partial\space Loss}{\partial\space W_{n-2}} \big)^2 
    + \beta_2 \big( \frac{\partial\space Loss}{\partial\space W_{n-1}} \big)^2 +
    \big( \frac{\partial\space Loss}{\partial\space W_{n}}\big)^2 \Big)
$$

ADAM controls the importance of momentum and RMS-Prop via $\beta_1$ and $\beta_2$.

### Loss Function Scaling Impact on ADAM

For SGD, by multiplying $0.5$ to loss function, the learning rate is reduced to half.
However, for ADAM, this is not true.

For $\frac{\partial\space Loss}{\partial\space W_{n}} \rightarrow m_{n+1}$ and $(\frac{\partial\space Loss}{\partial\space W_{n}})^2 \rightarrow v_{n+1}$, and finally for $\Delta W_{n+1} = \eta \frac{\hat{m}\_{n+1}}{\sqrt{\hat{v}\_{n+1}}+\epsilon}$, the square root operation cancels out the $0.5$ effect on the loss function.
In conclusion, scaling on loss function has no effect on ADAM learning/weight update.

## Bayesian Optimizer

