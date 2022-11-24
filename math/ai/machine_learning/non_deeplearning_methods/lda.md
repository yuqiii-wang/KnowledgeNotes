# Linear Discriminant Analysis (LDA)

LDA projects the source data space (assumed normal distributions) to a new space of lower dimensionality that tries clustering samples of the same labels.

The figure below shows a simple LDA classification that LD2 $2$d-$1$d projection renders a bad separation result while LD1's is much better. The $2$d-$1$d projection refers to the $2$-d data points projected onto the arrow-defined $1$-d lines.
 
This gives advantage over PCA that bases on SVD. The source data distribution is nearly a rectangle whose eigenvalues $\bold{\sigma}_j$ might not be distinctive. 

![lda](imgs/lda.png "lda")

Consider a set of observations $\bold{x} \in \mathbb{R}^{N\times d}$ ($N$ is the total number of samples and $d$ is the number of input space dimensions) and labels $\bold{y}$. LDA assumes that the conditional probability density functions $p(\bold{x}|\bold{y}=y_j)$ are multivariate normal distributions $N(\bold{\mu}_j, \bold{\Sigma}_j)$ on a lower dimension projection space.

There are $k$ classes such that $\bold{y}_i \in \bold{C}=[C_1, C_2, ..., C_k]$. Denote $N_j$ as the number of samples for the $j$-th class $C_j$, and $X_j$ as the sample set for the class $C_j$. Accordingly, $\bold{\mu}_j$ refers to the mean of $X_j$ and $\bold{\Sigma}_j$ for the sigma of $X_j$.

The mean $\bold{\mu}_j$ and variance $\bold{\Sigma}_j$ for each class $C_j$ can be computed as below.
$$
\begin{align*}
\bold{\mu}_j &= \frac{1}{N_j} \sum_{\bold{x}_i \in X_j} \bold{x}_i
\\
\bold{\Sigma}_j &= \sum_{\bold{x}_i \in X_j} (\bold{x}_i - \bold{\mu}_j)(\bold{x}_i - \bold{\mu}_j)^\text{T}
\end{align*}
$$

## $2$-d Binary Classification Example

$2$-d data binary classification can be described as having the input $\bold{x}=\big[ \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}_1, \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}_2, ..., \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}_i, ..., \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}_n \big]^\text{T}$, 
and their corresponding labels $\bold{y}_i \in \bold{C}=[C_1, C_2]$. 

Still consider this example. A good separation such as LD1 should have vastly different $\mu_1$ and $\mu_2$, and large $\Sigma_1, \Sigma_2$ (indicating two distinctive "lean" Gaussian distributions). LD2 in contrast, has $\mu_1 \approx \mu_2$ and small $\Sigma_1, \Sigma_2$ (indicating two overlapping "fat" Gaussian distributions).

In other words, the below expression should be maximized.
$$
\frac{(\mu_1-\mu_2)^2}{\Sigma_1^2+\Sigma_2^2}
$$

In LDA, $S_b=(\mu_1-\mu_2)^2$ is termed *Between class scatter* and $S_w=\Sigma_1^2+\Sigma_2^2$ is termed *Within class scatter*.

![lda](imgs/lda.png "lda")

Here, the problem becomes optimizing the $2$d-$1$d projection so that $S_b$ is maximized and $S_w$ is minimized. 

Define a projection matrix $W$. Projection on each data point to a lower dimension space can be expressed as $\bold{w}^\text{T}\bold{x}_i$.

The variance can be computed as below.
$$
\begin{align*}
    \sum_{j=1}^{C=2} \sum_{\bold{x} \in X_j}
    (\bold{w}^\text{T}\bold{x}-\bold{w}^\text{T}\bold{\mu}_j)^2 &=
    \sum_{j=1}^{C=2} \sum_{\bold{x} \in X_j}\big(
        \bold{w}^\text{T}\ (\bold{x}-\bold{\mu}_j)
    \big)^2
    \\ &=
    \sum_{j=1}^{C=2} \sum_{\bold{x} \in X_j}
    \bold{w}^\text{T} (\bold{x}-\bold{\mu}_j)(\bold{x}-\bold{\mu}_j)^\text{T} \bold{w}
    \\ &=
    \bold{w}^\text{T} \sum_{j=1}^{C=2}  \sum_{\bold{x} \in X_j}
    (\bold{x}-\bold{\mu}_j)(\bold{x}-\bold{\mu}_j)^\text{T} \bold{w}
    \\ &=
    \bold{w}^\text{T} \bold{\Sigma}
     \bold{w}
\end{align*}
$$

The optimization can be expressed as below
$$
arg \space \underset{W}{max} \space
\bold{J}(W)
=
\frac{W^\text{T} S_b W}{W^\text{T} S_w W}
=
\frac{\big|\big|
    \bold{w} \bold{\mu}_0 - \bold{w} \bold{\mu}_1
\big|\big|^2}{
    \bold{w}^\text{T} \Sigma_0 \bold{w} + 
    \bold{w}^\text{T} \Sigma_1 \bold{w} 
}
$$

The optimal $W^*$ where $\bold{J}$ reaches maximum can be computed by $\frac{\partial \bold{J}}{\partial W}=0$.

Before the derivative computation, should first perform normalization. Otherwise, there is an additional dimension to optimize: projection scaling (reference homography). However, projection scaling does not affect the result of $\bold{J}(W)$, since only projection direction is required.

Here for normalization, set $\big|\big|\bold{w}^\text{T} S_w \bold{w}\big|\big|=1$. Introduce Lagrange multiplier, there is
$$
\mathcal{L}(\bold{w})=
\bold{w}^\text{T} S_b \bold{w} - \lambda(\bold{w}^\text{T} S_w \bold{w}-1)
$$

Compute the Lagrange multiplier, there is

$$
\begin{align*}
    && 
    \frac{\partial \mathcal{L}}{\partial \bold{w}}
    &=
    2 S_b \bold{w} - 2 \lambda S_w \bold{w} = 0
    \\ \Rightarrow && 
    S_b \bold{w} &= \lambda S_w \bold{w}
    \\ \Rightarrow && 
    S_w^{-1} S_b \bold{w} &= \lambda \bold{w}
\end{align*}
$$

The above derivation reveals that $W$ is the eigenvectors of $S_w^{-1} S_b$.

Remember $S_b = (\bold{\mu_1}-\bold{\mu_2})(\bold{\mu_1}-\bold{\mu_2})^\text{T}$; substitute it into the above equation and set $\lambda_m = (\bold{\mu_1}-\bold{\mu_2})^\text{T} \bold{w}$. The interesting thing is that $\lambda_m$ is a scalar ($(\bold{\mu_1}-\bold{\mu_2})^\text{T} \in \mathbb{R}^{1 \times d}$ and $\bold{w} \in \mathbb{R}^{d \times 1}$), and $\lambda$ in Lagrange multiplier definition is a scalar as well. Since projection scaling does not affect projection direction, $\lambda_m$ and $\lambda$ can be cancelled out.

The optimal projection $W^*$ that only concerns about the projection direction can be computed as below.

$$
\begin{align*}
&&
S_w^{-1} S_b \bold{w} = 
S_w^{-1} (\bold{\mu_1}-\bold{\mu_2})(\bold{\mu_1}-\bold{\mu_2})^\text{T} \bold{w}
&=
\lambda \bold{w}
\\ \Rightarrow &&
S_w^{-1} (\bold{\mu_1}-\bold{\mu_2}) \lambda_m
&=
\lambda \bold{w}
\\ \Rightarrow &&
\bold{w} &= S_w^{-1} (\bold{\mu_1}-\bold{\mu_2}) 
\end{align*}
$$

Finally, the optimal projection is $\bold{w}^*=S_w^{-1} (\bold{\mu_1}-\bold{\mu_2})$,
and the prediction can be computed by $\hat{\bold{y}}_i=\bold{w}^{*\text{T}}\bold{x}_i$.

To determine the label being $C_1$ or $C_2$, a threshold $t$ can be set up. The most common thresholding is taking the mean of the two classes' centroids $t=\frac{\bold{w}^{*\text{T}}\bold{\mu}_1 + \bold{w}^{*\text{T}}\bold{\mu}_2}{2}$. Then labelling can be expressed as
$$
\hat{\bold{y}}_i = 
\left\{
    \begin{array}{cc}
        C_1 & \bold{w}^{*\text{T}}\bold{x}_i \ge t
        \\
        C_2 & \bold{w}^{*\text{T}}\bold{x}_i < t
    \end{array}
\right.
$$

## Multivariate Scenarios

For multivariate distributions such as $\bold{x}=\bigg[ \begin{bmatrix} x_1 \\ x_2 \\ \vdots  \\ x_d\end{bmatrix}_1, \begin{bmatrix} x_1 \\ x_2 \\ \vdots  \\ x_d \end{bmatrix}_2, ..., \begin{bmatrix} x_1 \\ x_2 \\ \vdots  \\ x_d \end{bmatrix}_i, ..., \begin{bmatrix} x_1 \\ x_2 \\ \vdots  \\ x_d \end{bmatrix}_N \bigg]^\text{T}$, 
and their corresponding labels $\bold{y}_i \in \bold{C}=[C_1, C_2, ..., C_k]$, the $S_b$ and $S_w$ can be computed as below.

![lda_multiDims](imgs/lda_multiDims.png "lda_multiDims")

* $S_w$'s computation first finds $\bold{\mu}_j=\frac{1}{N_j}\sum^{N_j}_{\bold{x}_i \in X_j} \bold{x}_i$, then computes $S_w$ by summing up all classes' variances.

$$
S_w = \sum_{j=1}^{C_k} \sum_{\bold{x}_i \in X_j}
(\bold{w}^\text{T}\bold{x}_i-\bold{w}^\text{T}\bold{\mu}_j)^2
$$

* $S_b$'s computation first calculates the global mean $\bold{\mu}=\frac{1}{N}\sum^N_{i=1} \bold{x}_i$, then sums up the respective variances of each class mean $\bold{\mu}_j$. The variance is not normalized so that $N_j$ is included ($S_w$ sums up all $\bold{x}_i \in X_j$ so that $S_b$ needs $N_j$ for each class $C_j$). 
  
$$
S_b = \sum_{j=1}^{C_k} N_j (\bold{\mu}_j - \bold{\mu})(\bold{\mu}_j - \bold{\mu})^\text{T}
$$

## Discussions

Consider this equation $S_w^{-1} S_b \bold{w} = \lambda \bold{w}$ that gives the optimal projection $\bold{w}^*$. The general method is to compute eigenvalues of $S_w^{-1} S_b$, then pick top few eigenvalue-corresponded eigenvectors $\bold{w}$ that together form the projection matrix $W$.

Since the global mean $\bold{\mu}$ is known, the last class's mean $\bold{\mu}_k$ can be linearly represented by the previous $k-1$ classes' means $a_0 \bold{\mu} + \sum_{j=1}^{k-1} a_j \bold{\mu}_j$, where $a_j$ is the coefficient for the class $C_j$. As a result, $S_b$ can at most have $rank(S_b)=k-1$, which indicates that $S_b$ at most have $k-1$ eigenvectors, and $W$ can at most project into a $k-1$ dimensional space.

### Drawbacks

* Projection only has $k-1$ dimensions
* Eigenvectors are not necessarily orthogonal