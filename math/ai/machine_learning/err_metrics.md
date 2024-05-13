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
L_{CE}&=
-\sum_i^C t_i \space \log(s_i) \\ &=
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

### Edit Distance

For two vector $\bold{u}$ and $\bold{v}$, *Edit distance* is a way of quantifying how dissimilar two strings are.

#### Hamming Distance

Hamming distance computes the difference between each corresponding **binary** element, then sum them up, typically used in one-hot encoded strings.

$$
\sum_i^n |u_i - v_i|
,\qquad
u_i \in \bold{u} \in \{0,1\}, \space v_i \in \bold{v} \in \{0,1\}
$$

#### Longest Common Subsequence (LCS) vs Longest Common Substring

Concatenate common Substrings by order (not necessarily consecutive) and derive Longest Common Subsequence (LCS).
The longest common substring is the longest consecutive char sequence.

$$
\begin{align*}
& \underbrace{A,\underbrace{B,C,D,E,F,G}_{\text{Longest Comm Sub-Str}},H,\underbrace{I,J,K},L,M,N}_{\text{Longest Comm Seq: }B,C,D,E,F,G,I,J,K} \\ \space \\
& \underbrace{X,\underbrace{B,C,D,E,F,G}_{\text{Longest Comm Sub-Str}},X,X,Y,Y,\underbrace{I,J,K},X,X,Y,Y}_{\text{Longest Comm Seq: }B,C,D,E,F,G,I,J,K}
\end{align*}
$$

#### Levenshtein Distance

Levenshtein distance generalizes CRUD operation complexity on how many CRUD operations to take to make one string same as the other one.

The formula is recursive comparing the front char between two strings then recursively loading the remaining chars in which compare the front char again and again; if two front chars are different from two strings, Levenshtein distance increments by $\text{Lev} \leftarrow \text{Lev}+1$, such that

$$
\text{Lev}(\bold{u}, \bold{v}) =
\left\{ \begin{array}{cc}
    |\bold{u}| & \text{if } |\bold{u}| = 0 \\
    |\bold{v}| & \text{if } |\bold{v}| = 0 \\
    \text{Lev}(\text{tail}(\bold{u}), \text{tail}(\bold{v})) & \text{if } \text{head}(\bold{u}) = \text{head}(\bold{v}) \\
    1+\min \left\{
        \begin{array}{c}
            \text{Lev}(\text{tail}(\bold{u}), \bold{v}) \\
            \text{Lev}(\bold{u}, \text{tail}(\bold{v})) \\
            \text{Lev}(\text{tail}(\bold{u}), \text{tail}(\bold{v}))
        \end{array}
        \right. & \text{Otherwise}
\end{array}
\right.
$$

where $\text{head}(...)$ represents the FIRST char of a string such that $\text{head}(\bold{x})=\{ x_1 \}$, and $\text{tail}(...)$ refers to the remaining of a string except for the first char $\text{tail}(\bold{x})=\{x_2, x_{3}, ..., x_{n}\}$.

For example,

* "beast" $\rightarrow$ "best" has $\text{Lev}$ of $1$ (one DELETE operation)
* "west" $\rightarrow$ "eat" has $\text{Lev}$ of $2$ (one DELETE and one one UPDATE operation)
* "abcdefg" $\rightarrow$ "bcdefgh" has $\text{Lev}$ of $2$ (one DELETE to the head and one INSERT to the end)

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

KLD denoted as $D_{KL}(P || Q)$ is a measure of how a prediction probability distribution $Q$ is different from actual probability distribution $P$.

$$
D_{KL}(P || Q) =
\sum_{x \in X} P(x) \log \Big( \frac{P(x)}{Q(x)} \Big)
$$

### Jensen-Shannon Divergence

In contrast to $D_{KL}$ that tests the prediction distribution $P$ against reference distribution $Q$, *Jensen-Shannon divergence* $D_{JS}$ uses geometric mean $\frac{Q+P}{2}$ as the reference distribution comparing between $P$ and $Q$.

This means neither $P$ nor $Q$ is a reference, instead, $D_{JS}$ treats both as reference by taking the "mean" distribution of $P$ and $Q$, and compare $P$ and $Q$ against this mean $\frac{Q+P}{2}$.

$$
D_{JS}(P || Q) = \frac{1}{2} D_{KL}\Big(Q || \frac{Q+P}{2} \Big) + \frac{1}{2} D_{KL}\Big(P || \frac{Q+P}{2} \Big)
$$

A general form of $D_{JS}$ is by replacing $\frac{Q+P}{2}$ with weighs $\bold{\pi}=\{ \pi_1, \pi_2, ..., \pi_n \}$ on distributions $\bold{P} = \{ P_1, P_2, ..., P_n \}$.

## Errors


### Root mean Square Deviation (RMSD)

For $n$ samples of pairs $\{ y_i, x_i \}$ for a system $f(.)$, RMSD can be computed by

$$
L = \sqrt{\frac{1}{n} \sum_{i=1}^n \big( y_i - f(x_i) \big)}
$$
