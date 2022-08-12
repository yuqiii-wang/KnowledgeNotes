# Activation Functions

## sigmoid

$$
sigmoid \space x
=
\frac{e^{x}}{e^x+1}
$$

$sigmoid$ maps input $(-\infty, +\infty)$ to output $(0,1)$. It is used to represent zero/full information flow. For example, in LSTM, it guards in/out/forget gates to permit data flow.

## Softmax

Defien a standard (unit) softmax function $\sigma: \mathbb{R}^K \rightarrow (0,1)^k$
$$
\sigma(\bold{z})_i
=
\frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}
$$
for $i=1,2,...,K$ and $\bold{z}=(z_1, z_2, ..., z_K)\in \mathbb{R}^K$

Now define $\bold{z}=\bold{x}^\text{T}\bold{w}$, there is
$$
softmax \space (y=j | \bold{x})
=
\frac{e^{\bold{x}^\text{T}\bold{w}_j}}{\sum^K_{k=1}e^{\bold{x}^\text{T}\bold{w}_k}}
$$

$softmax$ is often used in the final layer of a classifier network that outputs each class energy.

## reLU

$$
relu \space x
=
max(0, x)
$$

$relu$ retains a constant gradient regardless of the input $x$. For $sigmoid$, gradeint approaches to zero when $x \rightarrow +\infty$, however, for $relu$, gradient remain constant.

$sigmoid$ generates positive gradients despite $x<0$, which might wrongfully encourage weight updates, while $relu$ simply puts it zero.

$relu$ is easy in computation.

## tanh

$$
tanh \space x 
=
\frac{e^{2x-1}}{e^{2x+1}}
$$

$tanh$ maps input $(-\infty, +\infty)$ to output $(-1,1)$, that is good for features requiring both negative and positive gradients updating weights of neurons. 

$tanh$ is a good activation function to tackling vanishing gradient issues.