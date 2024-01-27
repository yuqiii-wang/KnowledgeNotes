# CNN

## Convolutional Layer

### Typical Computation Cost of A Convolutional Layer

* Filter kernel size: $k \times k$
* Image size divided by stride: $\frac{m \times n}{s \times s}$
* The number of filters $l$

Total: $(k \times k) \times \frac{m \times n}{s \times s} \times l$

### Convolution Forward and Back Propagation

Given an input image $X$ and a filter $F$, one forward pass of convolution is $O = X \otimes F$

$$
\begin{bmatrix}
    O_{11} & O_{12} \\
    O_{21} & O_{22}
\end{bmatrix} =
\begin{bmatrix}
    X_{11} & X_{12} & X_{13} \\
    X_{21} & X_{22} & X_{23} \\
    X_{31} & X_{32} & X_{33}
\end{bmatrix} \otimes
\begin{bmatrix}
    F_{11} & F_{12} \\
    F_{21} & F_{22}
\end{bmatrix}
$$

unfold the $\otimes$ operator, there are

$$
\begin{align*}
O_{11} &= X_{11} F_{11} + X_{12} F_{12} + X_{21} F_{21} + X_{22} F_{22} \\
O_{12} &= X_{12} F_{11} + X_{13} F_{12} + X_{22} F_{21} + X_{23} F_{22} \\
O_{21} &= X_{21} F_{11} + X_{22} F_{12} + X_{31} F_{21} + X_{32} F_{22} \\
O_{22} &= X_{22} F_{11} + X_{23} F_{12} + X_{32} F_{21} + X_{33} F_{22} \\
\end{align*}
$$

The back propagation of $F_{11}$ given loss $\mathcal{L}$ is

$$
\frac{\partial \mathcal{L}}{\partial F_{11}} =
\frac{\partial \mathcal{L}}{\partial O_{11}} X_{11} +
\frac{\partial \mathcal{L}}{\partial O_{12}} X_{12} +
\frac{\partial \mathcal{L}}{\partial O_{21}} X_{21} +
\frac{\partial \mathcal{L}}{\partial O_{22}} X_{22}
$$

## AlexNet

<div style="display: flex; justify-content: center;">
      <img src="imgs/alexnet.png" width="30%" height="60%" alt="alexnet" />
</div>
</br>