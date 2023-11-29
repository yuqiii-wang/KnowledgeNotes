# Error Metrics

A loss function is for a single training example (scaling individual error output). 
A cost function, on the other hand, is about the direct gaps/residuals between outputs and ground truth. 

## Regression Loss

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

The Huber loss combines the best properties of MSE and MAE. 
It is quadratic for smaller errors and is linear otherwise (and similarly for its gradient). 
It is identified by its delta parameter $\delta$:
$$
L_{\delta}(e)=
\left\{
    \begin{array}{c}
        \frac{1}{2}e^2 &\quad \text{for} |e|\le \delta
        \\
        \delta \cdot (|e|-\frac{1}{2}\delta) &\quad \text{otherwise}
    \end{array}
\right.
$$

### Cauchy Loss

Similar to Huber loss, but Cauchy loss provides smooth curve that error output grows significantly when small, grows slowly when large.

$$
L(e) = \log (1+e)
$$

## Classification Loss

### Categorical Cross-Entropy

$$
\begin{align*}
L_{CE}
&=
-\sum_i^C t_i \space \log(s_i)
\\ &=
-\sum_i^C t_i \space \log(\frac{e^{z_i}}{\sum^C_{j=1}e^{z_j}})
\end{align*}
$$
where $t_i$ is the ground truth (one-hot encoded) for a total of $C$ classes for prediction, and $s_i$ is the softmax score for the $i$-th class.

Cross-entropy outputs entropy of each class error, then sums them up.
The one-hot encoding of $t_i$ means the prediction error $t_i \space \log(\frac{e^{z_i}}{\sum^C_{j=1}e^{z_j}})$ has non-zero value for $z_i$ corresponds to the $i$-th class prediction $i=c$.

$$
t_i=
\left\{
    \begin{array}{c}
        1 &\quad i=c
        \\
        0 &\quad i \ne c
    \end{array}
\right.
$$

### Hinge Loss

$$
L(y)=
max(0, 1-t \cdot y)
$$
where $t=\pm 1$ and $y$ is the prediction score. For example, in SVM, $y=\bold{w}^\text{T}\bold{x}+b$, in which $(\bold{w}, b)$ is the hyperplane.

## Distance

### Euclidean Distance

For two vector $\bold{u}$ and $\bold{v}$, Euclidean distance is calculated as the square root of the sum of the squared differences between the two vectors.

$$
\sqrt{\sum_i^n (u_i - v_i)^2}
,\qquad
u_i \in \bold{u} \in \mathbb{R}, \space v_i \in \bold{v} \in \mathbb{R}
$$

### Hamming Distance

For two vector $\bold{u}$ and $\bold{v}$, Hamming distance compute the difference between each corresponding **binary** element, then sum them up, typically used in one-hot encoded strings.

$$
\sum_i^n |u_i - v_i|
,\qquad
u_i \in \bold{u} \in \{0,1\}, \space v_i \in \bold{v} \in \{0,1\}
$$

### Manhattan Distance

For two vector $\bold{u}$ and $\bold{v}$, Manhattan distance compute the difference between each corresponding **real** element, then sum them up.
It is often referred to as $\mathcal{L}_1$ norm error, or the sum absolute error and mean absolute error metric.

$$
\sum_i^n |u_i - v_i|
,\qquad
u_i \in \bold{u} \in \mathbb{R}, \space v_i \in \bold{v} \in \mathbb{R}
$$

### Minkowski Distance

For two vector $\bold{u}$ and $\bold{v}$, Minkowski distance is a generalization of the Euclidean and Manhattan distance measures and adds a parameter, called the “order” or $p$, that allows different distance measures to be calculated.

$$
\sum_i^n \Big( \big(|u_i - v_i|\big)^p \Big)^{\frac{1}{p}}
,\qquad
u_i \in \bold{u} \in \mathbb{R}, \space v_i \in \bold{v} \in \mathbb{R}
$$
where $p$ is the order parameter.

When $p$ is set to $1$, the calculation is the same as the Manhattan distance. 
When $p$ is set to $2$, it is the same as the Euclidean distance.
Intermediate values provide a controlled balance between the two measures.

### Kullback-Leibler Divergence (KLD)

KLD denoted as $D_{KL}(P || Q)$ is a measure of how one probability distribution $P$ is different from a second reference probability distribution $Q$.

$$
D_{KL}(P || Q) =
\sum_{x \in X} P(x) \log \Big( \frac{P(x)}{Q(x)} \Big)
$$

For example, there are three events labelled $x \in \{0,1,2\}$ that are assumed each event should have the same probability of occurrence (reference distribution $Q(x)=\frac{1}{3}$ regardless of input $x$).
However, having conducted $25$ trials, here sees different observations: $9$ for event $x=0$, $12$ for event $x=1$ and $4$ for event $x=2$.


||$x=0$|$x=1$|$x=2$|
|-|-|-|-|
|$P(x)$|$\frac{9}{25}$|$\frac{12}{25}$|$\frac{4}{25}$|
|$Q(x)$|$\frac{1}{3}$|$\frac{1}{3}$|$\frac{1}{3}$|

$$
\begin{align*}
    D_{KL}(P || Q) &=
    \sum_{x \in X} P(x) \log \Big( \frac{P(x)}{Q(x)} \Big)
\\ &=
    \frac{9}{25}\ln \Big( \frac{9/25}{1/3} 
    \Big) +
    \frac{12}{25}\ln \Big( \frac{12/25}{1/3} \Big) +
    \frac{4}{25}\ln \Big( \frac{4/25}{1/3} \Big)
\\ &\approx
    0.085
\end{align*}
$$

## Errors


### Root mean Square Deviation (RMSD)

For $n$ samples of pairs $\{ y_i, x_i \}$ for a system $f(.)$, RMSD can be computed by

$$
L = \sqrt{\frac{1}{n} \sum_{i=1}^n \big( y_i - f(x_i) \big)}
$$