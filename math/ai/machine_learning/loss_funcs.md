# Loss/Cost Func

A loss function is for a single training example. A cost function, on the other hand, is the average loss over the entire training dataset. 

## Regression

### Squared Error Loss

Squared Error loss for each training example, also known as L2 Loss, and the corresponding cost function is the Mean of these Squared Errors (MSE).

$$
L = (y - f(x))^2
$$

### Absolute Error Loss

Also known as L1 loss. The cost is the Mean of these Absolute Errors (MAE).

$$
L = | y - f(x) |
$$

### Huber Loss

The Huber loss combines the best properties of MSE and MAE. It is quadratic for smaller errors and is linear otherwise (and similarly for its gradient). It is identified by its delta parameter $\delta$:
$$
L_{\delta}(a)=
\left\{
    \begin{array}{c}
        \frac{1}{2}a^2 &\quad \text{for} |a|\le \delta
        \\
        \delta (|a|-\frac{1}{2}\delta) &\quad \text{otherwise}
    \end{array}
\right.
$$

## Classification

### Categorical Cross-Entropy

$$
\begin{align*}
L_{CE}
&=
-\sum_i^C t_i \space log(s_i)
\\ &=
-\sum_i^C t_i \space log(\frac{e^{z_i}}{\sum^C_{j=1}e^{z_j}})
\end{align*}
$$
where $t_i$ is the ground truth for a total of $C$ classes for prediction, and $s_i$ is the softmax score for the $i$-th class.

### Hinge Loss

$$
L(y)=
max(0, 1-t \cdot y)
$$
where $t=\pm 1$ and $y$ is the prediction score. For example, in SVM, $y=\bold{w}^\text{T}\bold{x}+b$, in which $(\bold{w}, b)$ is the hyperplane.