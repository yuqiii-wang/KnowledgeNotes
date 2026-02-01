# Multi-Task Training/Fine-Tuning

Due to limited hardware resources, a large model should be versatile handling a wide range of tasks, and should be trained with a variety of task data.
Multi-task data might be vastly different in sizes and formats, and they may give imbalanced losses in training.
Small loss tasks are likely under-trained for model tends to prioritize optimizing large loss task data.

## Gradient Norm

Gradient norm can be used to balance different task gradients to prevent from biased optimization where some particular tasks may see large gradients.

Compute loss per task $i$ and accordingly the gradients $||\nabla_{W}L_i||_2$

Compute the ratio $\gamma_i$ per task

$$
\gamma_i=\frac{||\nabla_{W}L_i||_2}{\frac{1}{N}\sum^N_{j=1}||\nabla_{W}L_j||_2}
$$

Rewrite the loss as the sum over the deviation from task loss average $\overline{\gamma}\_i$.

$$
L\_{gradNorm}=\sum_i |w_i\gamma_i-\alpha\overline{\gamma}\_i|
$$

where $\alpha$ is a hyper-parameter to control the aptitude of the deviation.

The drawback is expensive computation that every layer weights need recomputation for $L\_{gradNorm}$.

## Task Normalization

Instead of computing every layer gradient, *task normalization* proposes computing per-task loss of the final result.

For every task $i$, compute *Exponential Moving Average* (EMA) as training step $t$ progresses.

$$
EMA_i(t)=\beta EMA_i(t-1) + (1-\beta) L_i(t)
$$

where $\beta$ is a decaying factor, e.g., 0.9, used to control history weights.

EMA is used as a per-task normalization term to make sure per task loss does not aggressively change.

$$
L\_{taskNorm}=\sum^N_{i=1}w_i\cdot \frac{L_i(t)}{EMA_i(t)}
$$

It is useful when some tasks are fast converging while others are not.

## Focal Loss

Focal loss adjust/increase the weights $\gamma_i$ of hard-to-solve task loss to mitigate loss imbalance issue via modifying standard cross entropy.

The standard cross entropy at the $t$-th training step for binary classification (output either be $y=1$ or $y=0$) is defined as

$$
L\_{CE}=-\log\big(p(t)\big),\qquad
p(t)=\begin{cases}
    p & \text{if } y=1 \\
    1-p & \text{if } y=0 \\
\end{cases}
$$

Focal loss adjustment is

$$
L\_{focal}=\alpha \big(1-p(t)\big)^{\gamma}\log\big(p(t)\big)
$$

When

* $p(t)\rightarrow 1$, this is an easy classification task, and $\big(1-p(t)\big)^{\gamma}\rightarrow 0$, the loss is suppressed
* $p(t)\rightarrow 0$, this is a difficult classification task, and $\big(1-p(t)\big)^{\gamma}\rightarrow 1$, the loss is similar ot the standard cross entropy
