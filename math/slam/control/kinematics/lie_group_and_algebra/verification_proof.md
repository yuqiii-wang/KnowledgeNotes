# Lie Group and Lie Algebra Proof for $SO(3)$, $SE(3)$, and $Sim(3)$

## Verify that $SO(3)$, $SE(3)$, and $Sim(3)$ are Groups on Matrix Multiplication.

Define rotation ${R} \in \mathbb{R}^{3 \times 3}$, translation ${t} \in \mathbb{R}^{3}$ (together denoted as $[{R}|{t}] \in \mathbb{R}^{4 \times 4}$) and scale-variant rotation plus translation $\bold{\zeta} \in \mathbb{R}^{7}$. 

Recall the properties of a group, for $X , Y, Z \in G$

* Closure: $X \circ Y \in G$

* Associativity: $(X \circ Y) \circ Z = X \circ (Y \circ Z)$

* Identity: $EI=XI=X$

* Inverse: $X^{-1}X = XX^{-1} = I$

### $SO(3)$

For $R_1 R_2 R_3 = (R_1 R_2) R_3 = R_1 (R_2 R_3) \in \mathbb{R}^{3 \times 3}$, $SO(3)$ satisfies the closure and associativity requirements.

For $(R_1 R_2)(R_1 R_2)^\text{T}=R_1 R_2 R_2^\text{T} R_1^\text{T}=I$ and $I (R_1 R_2)=R_1 R_2$, $SO(3)$ satisfies the identity requirement.

For $det(R_1 R_2)=1$, $SO(3)$ is invertible.

### $SE(3)$

Define $[{R}|{t}] = \begin{bmatrix}        R & t \\        0 & 1    \end{bmatrix}$. There is
$$
\begin{align*}
[{R}|{t}]_1 [{R}|{t}]_2 &= 
\begin{bmatrix}
    R_1 & t_1 \\
    0 & 1    
\end{bmatrix}
\begin{bmatrix}
    R_2 & t_2 \\
    0 & 1    
\end{bmatrix}
\\ &= \begin{bmatrix}
    R_1 R_2 & R_1 t_2 + t_1 \\
    0 & 1
\end{bmatrix}
\end{align*}
$$

$R_1 R_2 \in \mathbb{R}^{3 \times 3}$ satisfies the four group constraints as already proven above and there is $R_1 t_2 + t_1 \in \mathbb{R}^3$. So that $[{R}|{t}]_1 [{R}|{t}]_2 \in \mathbb{R}^{4 \times 4}$.

For $[{R}|{t}]_1 [{R}|{t}]_2 [{R}|{t}]_3 = \big([{R}|{t}]_1 [{R}|{t}]_2 \big) [{R}|{t}]_3 = [{R}|{t}]_1 \big([{R}|{t}]_2 [{R}|{t}]_3 \big) \in \mathbb{R}^{4 \times 4}$, 
$SE(3)$ satisfies the closure and associativity property.

For $I [{R}|{t}]_1 [{R}|{t}]_2 = [{R}|{t}]_1 [{R}|{t}]_2$, $SE(3)$ satisfies the identity properties.

For $det([{R}|{t}]_1 [{R}|{t}]_2) \ne 0$, $SE(3)$ is invertible.

### $Sim(3)$

${S} = \begin{bmatrix}
        s{R} & {t} \\
        {0} & 1
\end{bmatrix}
\in \mathbb{R}^{4 \times 4}$ is added a scaling factor $s \in \mathbb{R}^1$ to rotation $R$. The rest are the same as that of $SE(3)$.

$$
\begin{align*}
S_1 S_2 &= 
\begin{bmatrix}
    s_1 R_1 & t_1 \\
    0 & 1    
\end{bmatrix}
\begin{bmatrix}
    s_2 R_2 & t_2 \\
    0 & 1    
\end{bmatrix}
\\ &= \begin{bmatrix}
    s_1 s_2 R_1 R_2 & s_1 R_1 t_2 + t_1 \\
    0 & 1
\end{bmatrix}
\end{align*}
$$

Similar to that of $SE(3)$, it should be easy to prove that $Sim(3)$ have the four group properties.

## Verify that $so(3)$ and $se(3)$ satisfy the requirements of Lie algebra.

Recall the four properties of Lie algebra:

* Closure $\forall X, Y \in \mathbb{V}; [X,Y] \in \mathbb{V}$

* Bilinear composition $\forall X,Y,Z \in \mathbb{V}; a,b \in \mathbb{F}$
there are
$$
\begin{align*}
[aX + bY, Z] &=
a[X, Z] + y[Y, Z],
\\
[Z, aX + bY] &= a[Z,X] + b[Z,Y]
\end{align*}
$$

* Reflective $\forall X \in \mathbb{V}; [X,X] = 0$ 

* Jacobi identity
$$
\begin{align*}
\forall X,Y,Z &\in \mathbb{V};
\\
[X, [Y,Z]] + [Y, [X,Z]] &+ [Z, [X,Y]] = 0
\end{align*}
$$

Recall the definitions of $so(3)$ and $se(3)$.
$$
\begin{align*}
so(3) &= 
\{
    \phi \in \mathbb{R}^3 \text{ or } {\Phi}^\wedge \in \mathbb{R}^{3 \times 3}
\}
\\
se(3) &= 
\bigg\{
    \xi = \begin{bmatrix}
        \rho \\
        \phi
    \end{bmatrix}
    \in \mathbb{R}^6,
    \rho \in \mathbb{R}^3 ,
    \phi \in so(3) ,
    {\Phi}^\wedge = \begin{bmatrix}
        \phi^\wedge & \rho \\
        0 & 0
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
\bigg\}
\end{align*}
$$

Here only prove for $so(3)$. The proof for $se(3)$ should be easy by replacing $\phi \in \mathbb{R}^3$ with $\xi = \begin{bmatrix}        \rho \\        \phi    \end{bmatrix}    \in \mathbb{R}^6$.

### Closure

Define $\phi_1, \phi_2 \in \mathbb{R}^3$, show proof of $so(3)$ closure $[\phi_1, \phi_2] \in \mathbb{R}^3$ :

$$
\begin{align*}
  [\phi_1, \phi_2] &= 
  ({\Phi}_1^\wedge {\Phi}_2^\wedge - {\Phi}_2^\wedge {\Phi}_1^\wedge)^\vee
  \\ &=
  \bigg(\begin{bmatrix}
      0 & -\varphi_{1,z} & \varphi_{1,y} \\
      \varphi_{1,z} & 0 & -\varphi_{1,x} \\
      -\varphi_{1,y} & \varphi_{1,x}    & 0
  \end{bmatrix}
  \begin{bmatrix}
      0 & -\varphi_{2,z} & \varphi_{2,y} \\
      \varphi_{2,z} & 0 & -\varphi_{2,x} \\
      -\varphi_{2,y} & \varphi_{2,x}    & 0
  \end{bmatrix}
  \\ & \quad -
  \begin{bmatrix}
      0 & -\varphi_{2,z} & \varphi_{2,y} \\
      \varphi_{2,z} & 0 & -\varphi_{2,x} \\
      -\varphi_{2,y} & \varphi_{2,x}    & 0
  \end{bmatrix}
  \begin{bmatrix}
      0 & -\varphi_{1,z} & \varphi_{1,y} \\
      \varphi_{1,z} & 0 & -\varphi_{1,x} \\
      -\varphi_{1,y} & \varphi_{1,x}    & 0
  \end{bmatrix}\bigg)^\vee
  \\ &=
  \bigg(\begin{bmatrix}
      -\varphi_{1,z}\varphi_{2,z} - \varphi_{1,z}\varphi_{2,z} & 
      \varphi_{1,y}\varphi_{2,x} & 
      \varphi_{1,z}\varphi_{2,x} \\
      \varphi_{1,x}\varphi_{2,y} & 
      -\varphi_{1,z}\varphi_{2,z} - \varphi_{1,x}\varphi_{2,x} & 
      \varphi_{1,z}\varphi_{2,y} \\
      \varphi_{1,x}\varphi_{2,z} & 
      \varphi_{1,y}\varphi_{2,z} & 
      -\varphi_{1,y}\varphi_{2,y} - \varphi_{1,x}\varphi_{2,x}
  \end{bmatrix}
  \\ &\quad - 
  \begin{bmatrix}
      -\varphi_{1,z}\varphi_{2,z} - \varphi_{1,z}\varphi_{2,z} & 
      \varphi_{1,x}\varphi_{2,y} & 
      \varphi_{1,x}\varphi_{2,z} \\
      \varphi_{1,y}\varphi_{2,x} & 
      -\varphi_{1,z}\varphi_{2,z} - \varphi_{1,x}\varphi_{2,x} & 
      \varphi_{1,y}\varphi_{2,z} \\
      \varphi_{1,z}\varphi_{2,x} & 
      \varphi_{1,z}\varphi_{2,y} & 
      -\varphi_{1,y}\varphi_{2,y} - \varphi_{1,x}\varphi_{2,x}
  \end{bmatrix}\bigg)^\vee
  \\ &=
  \begin{bmatrix}
      0 & 
      \varphi_{1,y}\varphi_{2,x} - \varphi_{1,x}\varphi_{2,y} & 
      \varphi_{1,z}\varphi_{2,x} - \varphi_{1,x}\varphi_{2,z} \\
      \varphi_{1,x}\varphi_{2,y} - \varphi_{1,y}\varphi_{2,x} & 
      0 & 
      \varphi_{1,z}\varphi_{2,y} - \varphi_{1,y}\varphi_{2,z} \\
      \varphi_{1,x}\varphi_{2,z} - \varphi_{1,z}\varphi_{2,x} & 
      \varphi_{1,y}\varphi_{2,z} - \varphi_{1,z}\varphi_{2,y} & 
      0
  \end{bmatrix}^\vee 
  \\ &= 
  \begin{bmatrix}
      \varphi_{1,y}\varphi_{2,z} - \varphi_{1,z}\varphi_{2,y} & 
      \varphi_{1,z}\varphi_{2,x} - \varphi_{1,x}\varphi_{2,z} &
      \varphi_{1,x}\varphi_{2,y} - \varphi_{1,y}\varphi_{2,x}
  \end{bmatrix}
  \in \mathbb{R}^3
\end{align*}
$$

### Bilinear Composition

Define $\phi_1, \phi_2, \phi_3 \in \mathbb{R}^3, a,b \in \mathbb{R}$. Show bilinear composition property.

$$
\begin{align*}
[a\phi_1 + b\phi_2, \phi_3] &= \big(
    (a\Phi^\wedge_1 + b\Phi^\wedge_2)\Phi^\wedge_3 - \Phi^\wedge_3(a\Phi^\wedge_1 + b\Phi^\wedge_2)
\big)^\vee
\\ &=
\big( 
    a\Phi^\wedge_1 \Phi^\wedge_3 + b\Phi^\wedge_2 \Phi^\wedge_3 -
    a\Phi^\wedge_3\Phi^\wedge_1 - b\Phi^\wedge_3\Phi^\wedge_2
\big)^\vee
\\ &=
a(\Phi^\wedge_1\Phi^\wedge_3 - \Phi^\wedge_3\Phi^\wedge_1)^\vee
+b(\Phi^\wedge_2\Phi^\wedge_3 - \Phi^\wedge_3\Phi^\wedge_2)^\vee
\\ &=
a[\phi_1, \phi_3] + b[\phi_2, \phi_3] 
\end{align*}
$$

Similarly, there is
$$
[\phi_3, a\phi_1 + b\phi_2] = 
a[\phi_3, \phi_1] + b[\phi_3, \phi_2] 
$$

### Reflective

Define $\phi \in \mathbb{R}^3$. Simply, there is
$$
[\phi, \phi] = [\Phi^\wedge\Phi^\wedge-\Phi^\wedge\Phi^\wedge]^\vee= 0
$$

### Jacobi identity

Define $\phi_1, \phi_2, \phi_3 \in \mathbb{R}^3$

$$
\begin{align*}
\big[\phi_1, [\phi_2, \phi_3]\big] &=
\big(
    \Phi^\wedge_1(\Phi^\wedge_2\Phi^\wedge_3 - \Phi^\wedge_3\Phi^\wedge_2)
    -
    (\Phi^\wedge_2\Phi^\wedge_3 - \Phi^\wedge_3\Phi^\wedge_2)\Phi^\wedge_1
\big)^\vee
\\ &=
    (\Phi^\wedge_1 \Phi^\wedge_2 \Phi^\wedge_3)^\vee
    - (\Phi^\wedge_1 \Phi^\wedge_3 \Phi^\wedge_2)^\vee
    - (\Phi^\wedge_2 \Phi^\wedge_3 \Phi^\wedge_1)^\vee
    + (\Phi^\wedge_3 \Phi^\wedge_2 \Phi^\wedge_1)^\vee
\\
\big[\phi_2, [\phi_3, \phi_1]\big] &=
\big(
    \Phi^\wedge_2(\Phi^\wedge_3\Phi^\wedge_1 - \Phi^\wedge_1\Phi^\wedge_3)
    -
    (\Phi^\wedge_3\Phi^\wedge_1 - \Phi^\wedge_1\Phi^\wedge_3)\Phi^\wedge_2
\big)^\vee
\\ &=
    (\Phi^\wedge_2 \Phi^\wedge_3 \Phi^\wedge_1)^\vee
    - (\Phi^\wedge_2 \Phi^\wedge_1 \Phi^\wedge_3)^\vee
    - (\Phi^\wedge_3 \Phi^\wedge_1 \Phi^\wedge_2)^\vee
    + (\Phi^\wedge_1 \Phi^\wedge_3 \Phi^\wedge_2)^\vee
\\
\big[\phi_3, [\phi_1, \phi_2]\big] &=
\big(
    \Phi^\wedge_3(\Phi^\wedge_1\Phi^\wedge_2 - \Phi^\wedge_2\Phi^\wedge_1)
    -
    (\Phi^\wedge_1\Phi^\wedge_2 - \Phi^\wedge_2\Phi^\wedge_1)\Phi^\wedge_3
\big)^\vee
\\ &=
    (\Phi^\wedge_3 \Phi^\wedge_1 \Phi^\wedge_2)^\vee
    - (\Phi^\wedge_3 \Phi^\wedge_2 \Phi^\wedge_1)^\vee
    - (\Phi^\wedge_1 \Phi^\wedge_2 \Phi^\wedge_3)^\vee
    + (\Phi^\wedge_2 \Phi^\wedge_1 \Phi^\wedge_3)^\vee
\end{align*}
$$

Sum them up, there is
$$
\big[\phi_1, [\phi_2, \phi_3]\big] + \big[\phi_2, [\phi_3, \phi_1]\big]  + \big[\phi_3, [\phi_1, \phi_2]\big] = 0
$$

## Show that $R e^{\bold{p}^\wedge} R = e^{(R\bold{p})^\wedge}$ (the adjoint property of $SO(3)$)

### First, Show $R\bold{p}^\wedge R = (R\bold{p})^\wedge$

$\forall \bold{r}_1, \bold{r}_2, \bold{r}_3 \in \mathbb{R}^3$, Define $R$ and $R^\text{T}$ such as
$$
R = \begin{bmatrix}
    \bold{r}_1^\text{T} \\
    \bold{r}_2^\text{T} \\
    \bold{r}_3^\text{T}
\end{bmatrix}
, \quad 
R^\text{T} = \begin{bmatrix}
    \bold{r}_1 &
    \bold{r}_2 &
    \bold{r}_3
\end{bmatrix}
$$

Besides, define $\bold{p}=[p_x, p_y, p_z]$, so that $\bold{p}^\wedge=\begin{bmatrix}     0 & -p_z & p_y \\    p_z & 0 & -p_x \\    -p_y & p_x & 0\end{bmatrix}$.

The expression of $(R\bold{p})^\wedge$ is
$$
\begin{align*}
&& 
R\bold{p} &= \begin{bmatrix}
  \bold{r}_1^\text{T} \bold{p} & \bold{r}_2^\text{T} \bold{p} & \bold{r}_3^\text{T} \bold{p}
\end{bmatrix}
\\ \Rightarrow &&
(R\bold{p})^\wedge &= \begin{bmatrix}
    0 & -\bold{r}_3^\text{T}\bold{p} & \bold{r}_2^\text{T}\bold{p} \\
    \bold{r}_3^\text{T}\bold{p} & 0 & -\bold{r}_1^\text{T}\bold{p} \\
    -\bold{r}_2^\text{T}\bold{p} & \bold{r}_1^\text{T}\bold{p} & 0
\end{bmatrix}
\end{align*}
$$

Compute $\bold{p}^\wedge \bold{r}_i$, there is

$$
\begin{align*}
\bold{p}^\wedge \bold{r}_i &= 
\begin{bmatrix}
    0 & -p_z & p_y \\
    p_z & 0 & -p_x \\
    -p_y & p_x & 0
\end{bmatrix}
\begin{bmatrix}
    r_{i,x} \\
    r_{i,y} \\
    r_{i,z} \\
\end{bmatrix}
\\ &=
- p_z r_{i,y} + p_y r_{i,z}
+ p_z r_{i,x} - p_x r_{i,z}
- p_y r_{i,x} + p_x r_{i,y}
\\ &=
r_{i,x} (p_z - p_y)
+ r_{i,y} (p_x - p_y)
+ r_{i,z} (p_y - p_x)
\\ &=
\bold{p} \times \bold{r}_i
\end{align*}
$$

So that,
$$
\begin{align*}
  R\bold{p}^\wedge R &= R (\bold{p}^\wedge R)
  \\ &=
  \begin{bmatrix}
    \bold{r}_1^\text{T} \\
    \bold{r}_2^\text{T} \\
    \bold{r}_3^\text{T}
  \end{bmatrix}
  \begin{bmatrix}
    \bold{p} \times \bold{r}_1 &
    \bold{p} \times \bold{r}_2 &
    \bold{p} \times \bold{r}_3
  \end{bmatrix}
  \\ &=
  \begin{bmatrix}
    \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_1) &
    \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_2) &
    \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_3) \\
    \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_1) &
    \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_2) &
    \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_3) \\
    \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_1) &
    \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_2) &
    \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_3)
  \end{bmatrix}
\end{align*}
$$

Recall the definition of vector dot product and cross product: define $\bold{a},\bold{b},\bold{c} \in \mathbb{R}^3$ and set $M=[\bold{a},\bold{b},\bold{c}]$, there is
$$
det(M) = \bold{a}^\text{T}(\bold{b}\times\bold{c})= \bold{b}^\text{T}(\bold{c}\times\bold{a})= \bold{c}^\text{T}(\bold{a}\times\bold{b})
$$

So that, set $M=[\bold{r}_i \quad \bold{p} \quad \bold{r}_i]$. Since $\bold{r}_i$ is perpendicular to $\bold{p}\times\bold{r}_i$, the dot product should be zero. 
$$
\begin{align*}
    det(M) &= 0 = \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_1) \\
    det(M) &= 0 = \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_2) \\
    det(M) &= 0 = \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_3) \\
\end{align*}
$$

Recall that a rotation matrix $R$ is orthogonal that $\bold{r}_1, \bold{r}_2, \bold{r}_3 \in \mathbb{R}^3$ are perpendicular to each other, hence, by vector cross product  right-hand rule, there are $\bold{r}_1 \times \bold{r}_3 = \bold{r}_2,\space\space \bold{r}_1 \times \bold{r}_2 = -\bold{r}_3,\space\space \bold{r}_2 \times \bold{r}_3 = -\bold{r}_1$.

This time, set $M=[\bold{r}_i \quad \bold{p} \quad \bold{r}_j], \quad \forall i \ne j$, there are
$$
\begin{align*}
    \bold{r}_1^\text{T}(\bold{p} \times \bold{r}_2) &=
    \bold{p}^\text{T}(\bold{r}_1 \times \bold{r}_2) = \bold{p}^\text{T} (-\bold{r}_3) =
    -\bold{r}_3^\text{T}\bold{p}
    \\
    \bold{r}_2^\text{T}(\bold{p} \times \bold{r}_1) &=
    \bold{p}^\text{T}(\bold{r}_2 \times \bold{r}_1) = \bold{p}^\text{T} \bold{r}_3 =
    \bold{r}_3^\text{T}\bold{p}
    \\
    \bold{r}_3^\text{T}(\bold{p} \times \bold{r}_2) &=
    \bold{p}^\text{T}(\bold{r}_3 \times \bold{r}_2) = \bold{p}^\text{T} (-\bold{r}_1) =
    -\bold{r}_1^\text{T}\bold{p}
    \\
    \bold{r}_2^\text{T}(\bold{p} \times \bold{r}_3) &=
    \bold{p}^\text{T}(\bold{r}_2 \times \bold{r}_3) = \bold{p}^\text{T} \bold{r}_1 =
    \bold{r}_1^\text{T}\bold{p}
    \\
    \bold{r}_3^\text{T}(\bold{p} \times \bold{r}_1) &=
    \bold{p}^\text{T}(\bold{r}_3 \times \bold{r}_1) = \bold{p}^\text{T} (-\bold{r}_2) =
    -\bold{r}_2^\text{T}\bold{p}
    \\
    \bold{r}_1^\text{T}(\bold{p} \times \bold{r}_3) &=
    \bold{p}^\text{T}(\bold{r}_1 \times \bold{r}_3) = \bold{p}^\text{T} \bold{r}_2 =
    \bold{r}_2^\text{T}\bold{p}
\end{align*}
$$

Finally, combine them together, there is
$$
\begin{align*}
R\bold{p}^\wedge R &= 
\begin{bmatrix}
    \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_1) &
    \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_2) &
    \bold{r}_1^\text{T} (\bold{p} \times \bold{r}_3) \\
    \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_1) &
    \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_2) &
    \bold{r}_2^\text{T} (\bold{p} \times \bold{r}_3) \\
    \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_1) &
    \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_2) &
    \bold{r}_3^\text{T} (\bold{p} \times \bold{r}_3)
\end{bmatrix}
\\ &=
\begin{bmatrix}
    0 &
    -\bold{r}_3^\text{T}\bold{p} &
    \bold{r}_2^\text{T}\bold{p} \\
    \bold{r}_3^\text{T}\bold{p} &
    0 &
    -\bold{r}_1^\text{T}\bold{p} \\
    -\bold{r}_2^\text{T}\bold{p} &
    \bold{r}_1^\text{T}\bold{p} &
    0
\end{bmatrix}
\\ &=
(R\bold{p})^\wedge
\end{align*}
$$

### Show Adjoint Property of $SO(3)$: $R e^{\bold{p}^\wedge}R^\text{T}=e^{(R\bold{p})^\wedge}$

Expand $R e^{\bold{p}^\wedge} R^\text{T}$ by Taylor series, and recall that $R$ is orthogonal having this property: $R^\text{T} R = R R^\text{T} = I$, there is
$$
\begin{align*}
R e^{\bold{p}^\wedge} R^\text{T} &= 
R \sum^{+\infty}_{i=n} \bigg( \frac{(\bold{p}^\wedge)^n}{n!}\bigg) R^\text{T}
\\ &=
R \sum^{+\infty}_{i=n} \bigg( \frac{\bold{p}^\wedge\bold{p}^\wedge\bold{p}^\wedge ... \bold{p}^\wedge}{n!}\bigg) R^\text{T}
\\ &=
R \sum^{+\infty}_{i=n} \bigg( \frac{(\bold{p}^\wedge R^\text{T})(R\bold{p}^\wedge R^\text{T}) (R\bold{p}^\wedge R^\text{T}) ... (R\bold{p}^\wedge)}{n!}\bigg) R^\text{T}
\\ &=
\sum^{+\infty}_{i=n} \bigg( \frac{(R\bold{p}^\wedge R^\text{T})(R\bold{p}^\wedge R^\text{T}) (R\bold{p}^\wedge R^\text{T}) ... (R\bold{p}^\wedge R^\text{T})}{n!}\bigg)
\\ &=
\sum^{+\infty}_{i=n} \frac{(R\bold{p}^\wedge R^\text{T})^n}{n!}
\\ &=
\sum^{+\infty}_{i=n} \frac{((R\bold{p})^\wedge)^n}{n!}
\\ &=
e^{(R\bold{p})^\wedge}
\end{align*}
$$

