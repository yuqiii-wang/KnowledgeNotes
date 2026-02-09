# Curves

## Spline

Just a polynomial trajectory. 

* Linear

$$
p(x)=ax + b
\quad a \ne 0
$$

* Quadratic

$$
p(x)=ax^2 + bx + c
\quad a \ne 0
$$

* Cubic
$$
p(x)=ax^3+bx^2+cx+d
\quad a \ne 0
$$

* Quartic

$$
p(x)=ax^4+bx^3+cx^2 + dx + e
\quad a \ne 0
$$

* Quintic

$$
p(x)=ax^5+bx^4+cx^3+dx^2 + ex + f
\quad a \ne 0
$$

* Sextic/Hexic

$$
p(x)=ax^6+bx^5+cx^4+dx^3 + ex^2 + fx + g
\quad a \ne 0
$$

## Bézier curve

A recursive definition for the Bézier curve of degree $n$ expresses it as a point-to-point linear combination (linear interpolation) of a pair of corresponding points in two Bézier curves of degree $n − 1$.

$$
\begin{align*}
\mathbf{B}_{\mathbf{P}_0}(t)&= 
\mathbf{P}_0 \\\\
\mathbf{B}(t) &=
\mathbf{B}_{\mathbf{P}_0 \mathbf{P}_1 \mathbf{P}_2 ... \mathbf{P}_n}(t)\\\\ &=
(1-t)\mathbf{B}_{\mathbf{P}_0 \mathbf{P}_1 \mathbf{P}_2 ... \mathbf{P}_{n-1}}(t)
+
t\mathbf{B}_{\mathbf{P}_1 \mathbf{P}_2 ... \mathbf{P}_{n}}(t)\\\\ &=
(1-t)^n \mathbf{P}_0 
+
\bigg(\begin{array}{c}
n \\\\
1
\end{array} \bigg)
(1-t)^{n-1} t \mathbf{P}_1+... +
\bigg(\begin{array}{c}
n \\\\
n-1
\end{array} \bigg)
(1-t) t^{n-1} \mathbf{P}_{n-1}
+
t^n \mathbf{P}_n\\\\ &=
\sum^n\_{k=0} 
\bigg(\begin{array}{c}
n \\\\
k
\end{array} \bigg)
(1-t)^k t^{n-k}
\mathbf{P}_k
\end{align*}
$$

For fourth-order curves one can construct intermediate points $\mathbf{Q}=[Q_0, Q_1, Q_2, Q_3]$ that describe linear/first-order Bézier curves, points $\mathbf{R}=[R_0, R_1, R_2]$ (computed from the moving $\mathbf{Q}$) that describe quadratic Bézier curves, and points $\mathbf{S}=[S_0, S_1]$ (computed from the moving $\mathbf{R}$) that describe cubic Bézier curves.

The linear combinations of the above points forms a fourth-order curve.


<div style="display: flex; justify-content: center;">
      <img src="imgs/bezier_4_big.svg.png" width="30%" height="30%" alt="bezier_4_big.svg" />
</div>
</br>

<div style="display: flex; justify-content: center;">
      <img src="imgs/bezier_4_big.gif" width="30%" height="30%" alt="bezier_4_big.gif" />
</div>
</br>

![Bézier_4_big.svg](imgs/bezier_4_big.svg.png "Bézier_4_big.svg")

![Bézier_4_big](imgs/bezier_4_big.gif "Bézier_4_big")
