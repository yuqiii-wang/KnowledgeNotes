# Error Metrics

* Cost vs Loss

A loss function is for a single training example (scaling individual error output).

A cost function, on the other hand, is about the direct gaps/residuals between outputs and ground truth.

## Regression Loss

### Squared Error Loss (L2)

Squared Error loss for each training example, also known as L2 Loss, and the corresponding cost function is the Mean of these Squared Errors (MSE).

$$
L = (y - f(x))^2
$$

### Absolute Error Loss (L1)

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
        \\\\
        \delta \cdot (|e|-\frac{1}{2}\delta) &\quad \text{otherwise}
    \end{array}
\right.
$$

### Cauchy Loss

Similar to Huber loss, but Cauchy loss provides smooth curve that error output grows significantly when small, grows slowly when large.

$$
L(e) = \log (1+e)
$$

### Discussion: L1 vs L2 in Training

Define $L_2$ Loss ($\mathbf{w}$ is the weights to optimize):

$$
\begin{align*}
&& L_2 = \frac{1}{2} \Big|\Big| \mathbf{y} - X\mathbf{w} \Big|\Big|^2_2
& = \frac{1}{2}\Big(\mathbf{y} - X\mathbf{w}\Big)^{\top} \Big(\mathbf{y} - X\mathbf{w}\Big) \\\\
\Rightarrow && &= \frac{1}{2} \mathbf{w}X^{\top}X \mathbf{w}^{\top} - 2\mathbf{y}^{\top} X \mathbf{w} + \mathbf{y}^{\top}\mathbf{y}
\end{align*}
$$

Define $L_1$ Loss ($\mathbf{w}$ is the weights to optimize):

$$
L_1=\Big|\Big| \mathbf{y} - X\mathbf{w} \Big|\Big|_1 =
\sum_{i} \Big| y_i - X w_i \Big|
$$

Discussions:

* $L_2$ is twice differentiable, hence smooth.
* $\nabla^2 L_2 = X^{\top}X$ must be positive definite so that $L_2$ is convex
* $L_1$ is NOT smooth, that means there is no $\nabla^2$ indicating how good an iterative step is; if $\nabla^2$ is present, when an iterative step is approaching the extreme/minimum, gradient is small, so that step stride should be set small as well.

$$
\frac{\partial |w_i|}{\partial w_i} = \begin{cases}
    1 & w_i > 0 \\\\
    -1 & w_i < 0 \\\\
    \text{undefined} & w_i = 0 \\\\
\end{cases}
$$

* $X^{\top}X$ must be positive definite so that $L_2$ is convex
* $L_2$ has wider convergence space (multi-dimension sphere), while $L_1$ has a squared diamond space

For example for 2D space, convergence space is $|w_1|+|w_2|\le c$ for $L_1$, $w_1^2 + w^2_2\le c$ is a circle for $L_2$.
This circle is enclosing the $L_1$'s squared diamond.

Here $c$ is the constraint to prevent $w_i$ get too large.

## Classification Loss

### Categorical Cross Entropy

$$
\begin{align*}
L_{CE}&=
-\sum_i^C t_i \space \log(s_i) \\\\ &=
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
        \\\\
        0 &\quad i \ne c
    \end{array}
\right.
$$

### Hinge Loss

$$
L(y)=
max(0, 1-t \cdot y)
$$

where $t=\pm 1$ and $y$ is the prediction score. For example, in SVM, $y=\mathbf{w}^\text{T}\mathbf{x}+b$, in which $(\mathbf{w}, b)$ is the hyperplane.

## Distance

### Euclidean Distance

For two vector $\mathbf{u}$ and $\mathbf{v}$, Euclidean distance is calculated as the square root of the sum of the squared differences between the two vectors.

$$
\sqrt{\sum_i^n (u_i - v_i)^2},\qquad
u_i \in \mathbf{u} \in \mathbb{R}, \space v_i \in \mathbf{v} \in \mathbb{R}
$$

### Edit Distance

For two vector $\mathbf{u}$ and $\mathbf{v}$, *Edit distance* is a way of quantifying how dissimilar two strings are.

#### Hamming Distance

Hamming distance computes the difference between each corresponding **binary** element, then sum them up, typically used in one-hot encoded strings.

$$
\sum_i^n |u_i - v_i|
,\qquad
u_i \in \mathbf{u} \in \{0,1\}, \space v_i \in \mathbf{v} \in \{0,1\}
$$

#### Longest Common Subsequence (LCS) vs Longest Common Substring

Concatenate common Substrings by order (not necessarily consecutive) and derive Longest Common Subsequence (LCS).
The longest common substring is the longest consecutive char sequence.

$$
\begin{align*}
& \underbrace{A,\underbrace{B,C,D,E,F,G}_{\text{Longest Comm Sub-Str}},H,\underbrace{I,J,K},L,M,N}_{\text{Longest Comm Sub-Seq: }B,C,D,E,F,G,I,J,K} \\\\ \space \\\\
& \underbrace{X,\underbrace{B,C,D,E,F,G}_{\text{Longest Comm Sub-Str}},X,X,Y,Y,\underbrace{I,J,K},X,X,Y,Y}_{\text{Longest Comm Sub-Seq: }B,C,D,E,F,G,I,J,K}
\end{align*}
$$

#### Levenshtein Distance

Levenshtein distance generalizes CRUD operation complexity on how many CRUD operations to take to make one string same as the other one.

The formula is recursive comparing the front char between two strings then recursively loading the remaining chars in which compare the front char again and again; if two front chars are different from two strings, Levenshtein distance increments by $\text{Lev} \leftarrow \text{Lev}+1$, such that

$$
\text{Lev}(\mathbf{u}, \mathbf{v}) =
\left\{ \begin{array}{cc}
    |\mathbf{u}| & \text{if } |\mathbf{u}| = 0 \\\\
    |\mathbf{v}| & \text{if } |\mathbf{v}| = 0 \\\\
    \text{Lev}(\text{tail}(\mathbf{u}), \text{tail}(\mathbf{v})) & \text{if } \text{head}(\mathbf{u}) = \text{head}(\mathbf{v}) \\\\
    1+\min \left\{
        \begin{array}{c}
            \text{Lev}(\text{tail}(\mathbf{u}), \mathbf{v}) \\\\
            \text{Lev}(\mathbf{u}, \text{tail}(\mathbf{v})) \\\\
            \text{Lev}(\text{tail}(\mathbf{u}), \text{tail}(\mathbf{v}))
        \end{array}
        \right. & \text{Otherwise}
\end{array}
\right.
$$

where $\text{head}(...)$ represents the FIRST char of a string such that $\text{head}(\mathbf{x})=\{ x_1 \}$, and $\text{tail}(...)$ refers to the remaining of a string except for the first char $\text{tail}(\mathbf{x})=\{x_2, x_{3}, ..., x_{n}\}$.

For example,

* "beast" $\rightarrow$ "best" has $\text{Lev}$ of $1$ (one DELETE operation)
* "west" $\rightarrow$ "eat" has $\text{Lev}$ of $2$ (one DELETE and one one UPDATE operation)
* "abcdefg" $\rightarrow$ "bcdefgh" has $\text{Lev}$ of $2$ (one DELETE to the head and one INSERT to the end)

### Manhattan Distance

For two vector $\mathbf{u}$ and $\mathbf{v}$, Manhattan distance compute the difference between each corresponding **real** element, then sum them up.
It is often referred to as $\mathcal{L}_1$ norm error, or the sum absolute error and mean absolute error metric.

$$
\sum_i^n |u_i - v_i|
,\qquad
u_i \in \mathbf{u} \in \mathbb{R}, \space v_i \in \mathbf{v} \in \mathbb{R}
$$

### Minkowski Distance

For two vector $\mathbf{u}$ and $\mathbf{v}$, Minkowski distance is a generalization of the Euclidean and Manhattan distance measures and adds a parameter, called the "order" or $p$, that allows different distance measures to be calculated.

$$
\sum_i^n \Big( \big(|u_i - v_i|\big)^p \Big)^{\frac{1}{p}}
,\qquad
u_i \in \mathbf{u} \in \mathbb{R}, \space v_i \in \mathbf{v} \in \mathbb{R}
$$

where $p$ is the order parameter.

When $p$ is set to $1$, the calculation is the same as the Manhattan distance.
When $p$ is set to $2$, it is the same as the Euclidean distance.
Intermediate values provide a controlled balance between the two measures.

### Cosine Distance

$$
\begin{align*}
    \text{CosineSimilarity}&=\frac{\mathbf{u}\mathbf{v}}{||\mathbf{u}||\space||\mathbf{v}||} \\\\
    \text{CosineDistance}&=1-\text{CosineSimilarity}
\end{align*}
$$

Cosine distance disregards vector length but only considers the angle.

#### Minkowski Distance vs Cosine Distance

For example, given two articles, they are describing the same topic but different in length, say one got 200 words, another one got 3000 words.
Having computed embeddings on all the words, the summed cosine distance should be much smaller than that from Minkowski distance.

### Kullback-Leibler Divergence (KLD)

KLD denoted as $D_{KL}(P || Q)$ is a measure of how a prediction probability distribution $Q$ is different from actual probability distribution $P$.

$$
D_{KL}(P || Q) =
\sum_{x \in X} P(x) \log \Big( \frac{P(x)}{Q(x)} \Big)
$$

#### Ratio-Format KL Divergence

Ratio-format KL divergence takes the ratio $\gamma=\frac{Q(x)}{P(x)}$ and decomposes the standard $D_{KL}(P || Q)$ into the sum of individual samples.
In training/optimization it reveals dynamics of differences among distribution, that is particularly useful on weight-adjusting scenarios.

$$
D_{KL}=\sum_{x \in X} P(x) \Big(\gamma-\log \gamma -1 \Big)=
\sum_{x \in X} P(x) \bigg(\frac{Q(x)}{P(x)}-\log\frac{Q(x)}{P(x)}-1\bigg)\ge 0
$$

The implication in training/optimization is that the ratio-format $D_{KL}(P || Q)$ computes $\gamma-\log\gamma-1$ which is exactly individual sample contribution to the divergence.
$P(x)$ is the weight.

If only one sample is considered rather than the whole distribution, one can simply define this.

$$
D_{KL}=\gamma-\log \gamma -1
$$

##### Ratio-Format KL Divergence Derivation

Given $D_{KL}(P || Q) = \sum_{x \in X} P(x) \log \Big( \frac{P(x)}{Q(x)} \Big)$,
introduce $\gamma=\frac{Q(x)}{P(x)}$,
so that $\log \frac{P(x)}{Q(x)}=-\log\gamma$, and take it into the $D_{KL}$, there is

$$
D_{KL}(P || Q) = \sum_{x \in X} P(x) \Big( -\log\gamma \Big)
$$

Given the inequality $\log\gamma\le\gamma-1$, there is

$$
\sum_{x \in X} P(x) \Big( -\log\gamma \Big) \ge
\sum_{x \in X} P(x) \Big( 1-\gamma \Big)
$$

where $\sum_{x \in X} P(x) \Big( 1-\gamma \Big)=\sum_{x \in X}(1-\frac{Q(x)}{P(x)})=\sum_{x \in X}(P(x)-Q(x))$ reveals individual sample gap $P(x)-Q(x)$ contribution the the sum over the sample space $\sum_{x \in X}$.

Finally,

$$
\begin{align*}
    & \sum_{x \in X} P(x) \Big( -\log\gamma \Big) \ge
    \sum_{x \in X} P(x) \Big( 1-\gamma \Big) \\\\
\Rightarrow\qquad & \sum_{x \in X} P(x) \Big( \gamma-\log\gamma-1 \Big) \ge 0
\end{align*}
$$

### Jensen-Shannon Divergence

In contrast to $D_{KL}$ that tests the prediction distribution $P$ against reference distribution $Q$, *Jensen-Shannon divergence* $D_{JS}$ uses geometric mean $\frac{Q+P}{2}$ as the reference distribution comparing between $P$ and $Q$.

This means neither $P$ nor $Q$ is a reference, instead, $D_{JS}$ treats both as reference by taking the "mean" distribution of $P$ and $Q$, and compare $P$ and $Q$ against this mean $\frac{Q+P}{2}$.

$$
D_{JS}(P || Q) = \frac{1}{2} D_{KL}\Big(Q || \frac{Q+P}{2} \Big) + \frac{1}{2} D_{KL}\Big(P || \frac{Q+P}{2} \Big)
$$

A general form of $D_{JS}$ is by replacing $\frac{Q+P}{2}$ with weighs $\mathbf{\pi}=\{ \pi_1, \pi_2, ..., \pi_n \}$ on distributions $\mathbf{P} = \{ P_1, P_2, ..., P_n \}$.

## Errors

### Root mean Square Deviation (RMSD)

For $n$ samples of pairs $\{ y_i, x_i \}$ for a system $f(.)$, RMSD can be computed by

$$
L = \sqrt{\frac{1}{n} \sum_{i=1}^n \big( y_i - f(x_i) \big)}
$$
