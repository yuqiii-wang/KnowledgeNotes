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

 

### Hinge Loss
