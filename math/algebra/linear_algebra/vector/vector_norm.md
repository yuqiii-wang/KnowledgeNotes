# Vector Normalization

## Common Normalizations

### L1 Norm Normalization

$$
\mathbf{v}\_{\text{norm}} = \frac{\mathbf{v}}{||\mathbf{v}||_1}
\qquad \text{where } ||\mathbf{v}||_1=\sum^n\_{i=1} |v_i|
$$

### L2 Norm (Euclidean Norm) Normalization

L2 normalization scales a vector so that its Euclidean norm (or L2 norm) is 1.
Used to transform vectors to unit vector (length of 1.).

$$
\mathbf{v}\_{\text{norm}} = \frac{\mathbf{v}}{||\mathbf{v}||_2}
\qquad \text{where } ||\mathbf{v}||_2=\sqrt{\sum^n\_{i=1} v_i^2}
$$

### Min-Max Normalization

Used when features need to be compared on a common scale.

$$
{v}\_{\text{norm}, i} = \frac{{v}\_i-\min(\mathbf{v})}{\max(\mathbf{v})-\min(\mathbf{v})}
$$

### Z-score Normalization (Standardization)

Used when the data follows a Gaussian distribution of a mean $\mu(\mathbf{v})$ and standard deviation $\sigma(\mathbf{v})$, that the normalized vectors $\mathbf{v}\_{\text{norm}}$ have a mean of 0 and a standard deviation of 1.

$$
{v}\_{\text{norm}, i} = \frac{{v}\_i-\mu(\mathbf{v})}{\sigma(\mathbf{v})}
$$

## Gram-Schmidt process

Gram-Schmidt process is a method for orthonormalizing a set of vectors in an inner product space.

Denote a projection operator from vector $\mathbf{v}$ onto $\mathbf{u}$:
$$
proj_{\mathbf{u}}(\mathbf{v})=
\frac{\langle\mathbf{u},\mathbf{v}\rangle}{\langle\mathbf{u},\mathbf{u}\rangle}\mathbf{u}
$$

where $\langle\mathbf{u},\mathbf{v}\rangle$ represents inner product operation.

$$
\begin{array}{cc}
    \mathbf{u}_1 = \mathbf{v}_1 & 
    \mathbf{e}_1=\frac{\mathbf{u}_1}{||\mathbf{u}_1||}
    \\\\
    \mathbf{u}_2 = \mathbf{v}_2 - proj_{\mathbf{u}_1}(\mathbf{v}_2) & 
    \mathbf{e}_2=\frac{\mathbf{u}_2}{||\mathbf{u}_2||}
    \\\\
    \mathbf{u}_3 = \mathbf{v}_3 - proj_{\mathbf{u}_1}(\mathbf{v}_3) - proj_{\mathbf{u}_2}(\mathbf{v}_3) & 
    \mathbf{e}_3=\frac{\mathbf{u}_3}{||\mathbf{u}_3||}
    \\\\
    \mathbf{u}_4 = \mathbf{v}_4 - proj_{\mathbf{u}_1}(\mathbf{v}_4) - proj_{\mathbf{u}_2}(\mathbf{v}_4) - proj_{\mathbf{u}_3}(\mathbf{v}_4) & 
    \mathbf{e}_4=\frac{\mathbf{u}_4}{||\mathbf{u}_4||}
    \\\\
    \space
    \\\\
    ... & ...
    \\\\
    \space
    \\\\\
    \mathbf{u}_k = \mathbf{v}_k - \sum^{k-1}\_{j}proj_{\mathbf{u}_j}(\mathbf{v}_k) &
    \mathbf{e}_k=\frac{\mathbf{u}_k}{||\mathbf{u}_k||}
\end{array}
$$

<div style="display: flex; justify-content: center;">
      <img src="imgs/Gram-Schmidt_process.svg.png" width="40%" height="20%" alt="Gram-Schmidt_process.svg" />
</div>
</br>