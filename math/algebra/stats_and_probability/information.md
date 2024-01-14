# Information

## Entropy

The Shannon entropy $\text{H}$, in units of bits (per symbol), is given by

$$
\text{H} = - \sum_{i} p_i \log p_i
$$

where $p_i$ is the probability of occurrence of the $i$-th possible value of the source symbol.
Usually, $\log$ has a base of $2$ or Euler's number $e$.

* Joint Entropy

Given two random variables $X$ and $Y$, if $X$ and $Y$ are independent, their joint entropy is the sum of their individual entropies.

$$
\text{H}(X,Y) = -\sum_{x \in X, y \in Y} p(x,y) \log p (x,y)
$$

* Cross Entropy

Cross entropy $\text{H}$ measures how close two distributions $P$ and $Q$ are.
Set $P$ as the label truth token sequence, and $Q$ as the LLM prediction token sequence, so that predictions vs labels can be measured in cross entropy.

$$
\begin{align*}
\text{H}(P, Q) &= E_P \big( -\log Q \big) \\
    &= -\sum_{x \in \bold{x}} P(x) \log Q(x) \\
    &= -\sum_{x \in \bold{x}} P(x) \big(\log P(x) +  \log Q(x) -\log P(x) \big) \\
    &= -\sum_{x \in \bold{x}} P(x) \log P(x) - \sum_{x \in \bold{x}} P(x) \log \frac{Q(x)}{P(x)} \\
    &= \text{H}(P) + D_{KL}(P || Q)
\end{align*}
$$

where $D_{KL}(P || Q) > 0$ is Kullback-Leibler (KL) divergence describing how far between $P$ and $Q$. 

* Conditional entropy (equivocation)

Conditional uncertainty of $X$ given random variable $Y$ is the average conditional entropy over $Y$.

$$
\begin{align*}
\text{H}(X|Y) &= \mathbb{E}_Y \big( \text{H}(X|y) \big) 
\\ &=
\sum_{x \in Y} p(y) \sum_{x \in X} p(x|y) \log p(x|y)
\\ &=
-\sum_{x \in X, y \in Y} p(x,y) \log p (x|y)
\end{align*}
$$

* Mutual information (trans-information)

Mutual information measures the amount of information that can be obtained about one random variable by observing another.

$$
I(X;Y) = \sum_{x \in X, y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

If $X$ and $Y$ are totally independent, there is $p(x,y)=p(x)p(y)$, hence $\log \frac{p(x,y)}{p(x)p(y)}=0$.
If $X$ and $Y$ are co-dependent, there is $p(x,y)>p(x)p(y)$, hence $\log \frac{p(x,y)}{p(x)p(y)}>0$.

## Fisher Information

### Score

In statistics, score (or informant) $s(\theta)$ is a $1 \times m$ vector of the gradient of a log-likelihood $log L(\theta)$ function with respect to its config parameter vector $\theta$ (size of $1 \times m$). 

$$
s(\theta)=\frac{\partial \space log L(\theta)}{\partial \theta}
$$

### Mean

Given observations $\bold{x}=[x_1, x_2, ..., x_n]$ in the sample space $X$, the expected value of score is
$$
\begin{align*}
E(s|\theta)&=
\int_X \frac{\partial \space log L(\theta;x)}{\partial \theta} f(x;\theta) dx
\\ &=
\int_X \frac{\partial \space f(x; \theta)}{\partial \theta} \frac{1}{f(x;\theta)} f(x;\theta) dx
\\ &=
\int_X \frac{\partial \space f(x; \theta)}{\partial \theta} dx
\end{align*}
$$

By Leibniz integral rule which allows for interchange of derivative and integral, there is
$$
\begin{align*}
E(s|\theta) &=
\int_X \frac{\partial \space f(x; \theta)}{\partial \theta} dx
\\ &=
\frac{\partial }{\partial \theta} \int_X f(x;\theta) dx
\\ &=
\frac{\partial }{\partial \theta} 1 
\\ &= 
0
\end{align*}
$$

#### Intuition

The optimal configuration $\theta$ to fit sample space distribution is by minimizing its log-likelihood function $logL(\theta;x)$. 

$logL(\theta;x)$ by ideal $\theta$ should see its minima along side with its derivative zero, hence $E(s|\theta)=0$.


Fisher information is a way of measuring the amount of information that an observable random variable $\bold{x} \in X$ carries about an unknown parameter $\theta$.

Let $f(X;\theta)$ be the probability density function for $\bold{x} \in X$ conditioned on $\theta$.

Fisher information $\bold{I}(\theta)$ is defined to be the variance of score:
$$
\begin{align*}
\bold{I}(\theta)&=
E\bigg[
    \bigg(
        \frac{\partial \space log L(\bold{x};\theta)}{\partial \theta}  
    \bigg)^2
    \bigg| \theta
\bigg]
\\ &=
\int_X \bigg( \frac{\partial \space log L(\theta;x)}{\partial \theta} \bigg)^2 f(x;\theta) dx
\end{align*}
$$

### Twice differentiable with respect to $\theta$

For $\bold{I}(\theta)$ being twice differentiable with repssect to $\theta$, $\bold{I}(\theta)$ can be expressed as

$$
\begin{align*}
\bold{I}(\theta)&=
-E\bigg[
        \frac{\partial^2 \space log L(\bold{x};\theta)}{\partial \theta^2}  
    \bigg| \theta
\bigg]
\end{align*}
$$

Thus, the Fisher information may be seen as the curvature of the support curve (the graph of the log-likelihood).

### Fisher information matrix

For a multi-dimensional observation vector $x_k=(x_{k,1}, x_{k,2}, ..., x_{k,l})$ in the dataset $[x_1, x_2, ..., x_k, ..., x_n] \in \bold{x}$, a Fisher information matrix is the covariance of score, in which each entry is defined
$$
\bold{I}_{i,j}(\theta)=
E\bigg[
    \bigg(\frac{\partial \space log\space f(\bold{x};\theta)}{\partial \theta_i}\bigg)
    \cdot
    \bigg(\frac{\partial \space log\space f(\bold{x};\theta)}{\partial \theta_j}\bigg)^\text{T}
    \bigg| \theta
\bigg]
$$

### Intuition

Fisher information is defined by computing the covariance of the gradient of log-likelihood function $log L(\theta)$.

$\bold{I}_{i,j}(\theta)$ says that, the greater the one covariance element value, the greater the gradient of the partial derivative direction $\angle (\theta_i +  \theta_j)$, indicating optimization directions and volumes.