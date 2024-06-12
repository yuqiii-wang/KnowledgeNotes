# Vector Normalization

## Common Normalizations

### L1 Norm Normalization

$$
\bold{v}_{\text{norm}} = \frac{\bold{v}}{||\bold{v}||_1}
\qquad \text{where } ||\bold{v}||_1=\sum^n_{i=1} |v_i|
$$

### L2 Norm (Euclidean Norm) Normalization

L2 normalization scales a vector so that its Euclidean norm (or L2 norm) is 1.
Used to transform vectors to unit vector (length of 1.).

$$
\bold{v}_{\text{norm}} = \frac{\bold{v}}{||\bold{v}||_2}
\qquad \text{where } ||\bold{v}||_2=\sqrt{\sum^n_{i=1} v_i^2}
$$

### Min-Max Normalization

Used when features need to be compared on a common scale.

$$
{v}_{\text{norm}, i} = \frac{{v}_i-\min(\bold{v})}{\max(\bold{v})-\min(\bold{v})}
$$

### Z-score Normalization (Standardization)

Used when the data follows a Gaussian distribution of a mean $\mu(\bold{v})$ and standard deviation $\sigma(\bold{v})$, that the normalized vectors $\bold{v}_{\text{norm}}$ have a mean of 0 and a standard deviation of 1.

$$
{v}_{\text{norm}, i} = \frac{{v}_i-\mu(\bold{v})}{\sigma(\bold{v})}
$$

## Gram-Schmidt process

Gram-Schmidt process is a method for orthonormalizing a set of vectors in an inner product space.

Denote a projection operator from vector $\bold{v}$ onto $\bold{u}$:
$$
proj_{\bold{u}}(\bold{v})=
\frac{\langle\bold{u},\bold{v}\rangle}{\langle\bold{u},\bold{u}\rangle}\bold{u}
$$
where $\langle\bold{u},\bold{v}\rangle$ represents inner product operation.

$$
\begin{array}{cc}
    \bold{u}_1 = \bold{v}_1 & 
    \bold{e}_1=\frac{\bold{u}_1}{||\bold{u}_1||}
    \\
    \bold{u}_2 = \bold{v}_2 - proj_{\bold{u}_1}(\bold{v}_2) & 
    \bold{e}_2=\frac{\bold{u}_2}{||\bold{u}_2||}
    \\
    \bold{u}_3 = \bold{v}_3 - proj_{\bold{u}_1}(\bold{v}_3) - proj_{\bold{u}_2}(\bold{v}_3) & 
    \bold{e}_3=\frac{\bold{u}_3}{||\bold{u}_3||}
    \\
    \bold{u}_4 = \bold{v}_4 - proj_{\bold{u}_1}(\bold{v}_4) - proj_{\bold{u}_2}(\bold{v}_4) - proj_{\bold{u}_3}(\bold{v}_4) & 
    \bold{e}_4=\frac{\bold{u}_4}{||\bold{u}_4||}
    \\
    \space
    \\
    ... & ...
    \\
    \space
    \\\
    \bold{u}_k = \bold{v}_k - \sum^{k-1}_{j}proj_{\bold{u}_j}(\bold{v}_k) &
    \bold{e}_k=\frac{\bold{u}_k}{||\bold{u}_k||}
\end{array}
$$

<div style="display: flex; justify-content: center;">
      <img src="imgs/Gram-Schmidt_process.svg.png" width="40%" height="20%" alt="Gram-Schmidt_process.svg" />
</div>
</br>