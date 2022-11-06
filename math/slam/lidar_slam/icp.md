# Iterative Closest Point (ICP)

Given some source points, with applied transformation (rotation and translation $[\bold{R}|\bold{t}]$), algorithm iteratively revises the transformation $[\bold{R}|\bold{t}]$ to minimize an error metric $\bold{e}$ (typically the sum of squared differences) between output (result of the transformation) point cloud and reference (observed) cloud.

Suppose there are two sets of to-be-matched 3D points; the correspondences are shown as below
$$
\bold{P}=[\bold{p}_1, \bold{p}_2, ..., \bold{p}_n]
,\quad
\bold{P}'=[\bold{p}'_1, \bold{p}'_2, ..., \bold{p}'_n]
$$

There should be extrinsics (rotation and translation) that corresponds the two sets of points.
$$
\begin{align*}
\forall i, \bold{p}_i &= \bold{R}\bold{p}'_i+\bold{t}
\\ &=
[\bold{R}|\bold{t}] \bold{p}'_i
\end{align*}
$$

The error $\bold{e}_i$ can be defined the difference between the base point $\bold{p}_i$ and the correspondence point $\bold{p}'_i$
$$
\bold{e}_i = 
\bold{p}_i - (\bold{R}\bold{p}'_i + \bold{t})
$$

Accordingly, the least-square problem to find the optimal $[\bold{R}|\bold{t}]^*$ with minimized error $\bold{e}$ can be defined as

$$
\begin{align*}
\big[\bold{R}|\bold{t} \big]^* 
&= 
arg \space \underset{\big[\bold{R}|\bold{t} \big]}{min}
\frac{1}{2} \sum^n_{i=1}
\big|\big|
    \bold{e}_i
\big|\big|^2_2
\\ &=
arg \space \underset{\big[\bold{R}|\bold{t} \big]}{min}
\frac{1}{2} \sum^n_{i=1}
\bigg|\bigg|
  \bold{p}_i -  \big[\bold{R}|\bold{t} \big] \bold{p}'_i
\bigg|\bigg|^2_2
\\ &=
arg \space \underset{\big[\bold{R}|\bold{t} \big]}{min}
\frac{1}{2} \sum^n_{i=1}
\bigg|\bigg|
  \bold{p}_i -  (\bold{R} \bold{p}'_i + \bold{t})
\bigg|\bigg|^2_2
\end{align*}
$$

Define the centroids $\bold{p}_c, \bold{p}'_c$ of the point clouds $\bold{P}$ and $\bold{P}'$, respectively, there are
$$
\bold{p}_c = \frac{1}{n} \sum_{i=1}^n \bold{p}_i
, \quad
\bold{p}'_c = \frac{1}{n} \sum_{i=1}^n \bold{p}'_i
$$

Compute the least square function with the centroids such as
$$
\begin{align*}
    \frac{1}{2} \sum^n_{i=1}
    \bigg|\bigg|
        \bold{p}_i -  \big[\bold{R}|\bold{t} \big] \bold{p}'_i
    \bigg|\bigg|^2_2
    &=
    \frac{1}{2} \sum^n_{i=1}
    \bigg|\bigg|
        \bold{p}_i - (\bold{R} \bold{p}'_i + \bold{t})
    \bigg|\bigg|^2_2
    \\ &=
    \frac{1}{2} \sum^n_{i=1}
    \bigg|\bigg|
        \bold{p}_i - \bold{R} \bold{p}'_i - \bold{t} 
        - \bold{p}_c + \bold{p}_c
        - \bold{R}\bold{p}'_c + \bold{R}\bold{p}'_c
    \bigg|\bigg|^2_2
    \\ &=
    \frac{1}{2} \sum^n_{i=1}
    \bigg|\bigg|
        \big(\bold{p}_i - \bold{p}_c -\bold{R}(\bold{p}'_i - \bold{p}'_c)\big)
        +
        (\bold{p}_c-\bold{R}\bold{p}'_c-\bold{t})
    \bigg|\bigg|^2_2
    \\ &=
    \frac{1}{2} \sum^n_{i=1}
    \bigg(
        \big|\big| 
            \bold{p}_i - \bold{p}_c -\bold{R}(\bold{p}'_i - \bold{p}'_c)
        \big|\big|^2_2
        \\ &\quad\quad\quad +
        \big|\big| 
            \bold{p}_c-\bold{R}\bold{p}'_c-\bold{t}
        \big|\big|_2^2
        \\ &\quad\quad\quad +
        \underbrace{        
            2\big(\bold{p}_i - \bold{p}_c -\bold{R}(\bold{p}'_i - \bold{p}'_c)\big)^\text{T} (\bold{p}_c-\bold{R}\bold{p}'_c-\bold{t})
        }_{\begin{matrix}
            =0 \\
            \text{because} \sum^n_{i=1}\bold{p}_c-\sum^n_{i=1}\bold{p}_i=0
            \\ \text{ and }
            \sum^n_{i=1}\bold{p}'_c-\sum^n_{i=1}\bold{p}'_i=0
        \end{matrix}
        }
    \bigg)
    \\ &=
    \frac{1}{2} \sum^n_{i=1}
    \bigg(
        \big|\big| 
            \bold{p}_i - \bold{p}_c -\bold{R}(\bold{p}'_i - \bold{p}'_c)
        \big|\big|^2_2
        \\ &\quad\quad\quad +
        \big|\big| 
            \bold{p}_c-\bold{R}\bold{p}'_c-\bold{t}
        \big|\big|_2^2
    \bigg)
\end{align*}
$$

The first term $\big|\big| \bold{p}_i - \bold{p}_c -\bold{R}(\bold{p}'_i - \bold{p}'_c)\big|\big|^2_2$ only contains a rotation matrix $\bold{R}$, and the second term $\big|\big| \bold{p}_c-\bold{R}\bold{p}'_c-\bold{t} \big|\big|_2^2$ only has the centroids $\bold{p}_c$ and $\bold{p}'_c$. The $\bold{t}$ in the second term can be solved if $\bold{R}$ is known. The $\bold{R}$ can be solved in the first term by setting $\bold{q}_i = \bold{p}_i - \bold{p}_c$ and $\bold{q}'_i = \bold{p}'_i - \bold{p}'_c$, so that the optimal rotation matrix $\bold{R}^*$ can be computed as
$$
\bold{R}^*=
arg \space \underset{\bold{R}}{min}
\frac{1}{2} \sum^n_{i=1}
\bigg|\bigg|
  \bold{q}_i - \bold{R} \bold{q}'_i
\bigg|\bigg|^2_2
$$

The least square problem for $\bold{R}^*$ can be rewritten as
$$
\frac{1}{2} \sum^n_{i=1}
\bigg|\bigg|
  \bold{q}_i - \bold{R} \bold{q}'_i
\bigg|\bigg|^2_2
=
\frac{1}{2} \sum^n_{i=1}
\bigg(
    \underbrace{\bold{q}^\text{T}_i \bold{q}_i}_{\frac{\partial \bold{q}^\text{T}_i \bold{q}_i}{\partial \bold{R}}=0}
    +
    \bold{q}^\text{T}_i \underbrace{\bold{R}^\text{T}\bold{R}}_{=\bold{I}} \bold{q}'_i
    -
    \underbrace{2\bold{q}^\text{T}_i \bold{R} \bold{q}'_i}_{\text{To be maximized}}
\bigg)
$$

So that, finding $\bold{R}^*$ only needs to take care of this to-be-maximized term (here defines $\bold{q}^\text{T}_i \bold{R} \bold{q}'_i$ as the diagonal entries of $\bold{Q}^\text{T} \bold{R} \bold{Q}'$). By introducing trace operation, here derives:
$$
\begin{align*}
    \frac{1}{2} \sum^n_{i=1}
    2\bold{q}^\text{T}_i \bold{R} \bold{q}'_i
    &=
    tr \big(
        \bold{Q}^\text{T} \bold{R} \bold{Q}'
    \big)
    \\ &=
    tr \big(
        \bold{R} \bold{Q}' \bold{Q}^\text{T}
    \big)
\end{align*}
$$

Take SVD decomposition of $\bold{Q}' \bold{Q}^\text{T}$, there is
$$
\bold{Q}' \bold{Q}^\text{T}
=
\bold{U} \bold{\Sigma} \bold{V}^\text{T}
$$

Then consider the trace operator.
$$
tr \big(
        \bold{R} \bold{Q}' \bold{Q}^\text{T}
\big)
=
tr \big(
        \bold{R} \bold{U} \bold{\Sigma} \bold{V}^\text{T}
\big)
=
tr \big(
        \bold{\Sigma} 
        \underbrace{\bold{V}^\text{T} \bold{R} \bold{U}}_{=\bold{M}}
\big)
$$

Set $\bold{M}=\bold{V}^\text{T} \bold{R} \bold{U}$. Note that $\bold{V}^\text{T}, \bold{R}, \bold{U}$ are orthogonal. Therefore, $\bold{M}$ is orthogonal as well (each column vector $\bold{m}_j \in \bold{M}$ is an orthonormal vector).

Due to orthogonality, there is $\bold{M}^\text{T}\bold{M}=\bold{I}$. So that for each column vector, the vector product should be $1$.
Set $d=3$ for the rigid motion $SE(3)$, and derived $|m_{ij}| \le 1$.
$$
\begin{align*}
&
\bold{m}_j^\text{T} \bold{m}_j =
\sum^d_{i=1} m_{ij}^2 = 
1
\\ \Rightarrow \quad & \quad\quad
0 \le m^2_{ij} \le 1
\\ \Rightarrow \quad & \quad\quad
0 \le |m_{ij}| \le 1
\end{align*}
$$

Generalize each vector's result to the whole matrix, there is
$$
\begin{align*}
tr\big(\bold{\Sigma M}\big)
&=
\begin{bmatrix}
    \sigma_1 & 0 & & 0 \\
    0 & \sigma_2 & & 0 \\
    & & \ddots & \\
    0 & 0 & & \sigma_d
\end{bmatrix}   
\begin{bmatrix}
    m_{11} & m_{12} & & m_{1d} \\
    m_{21} & m_{22} & & m_{2d} \\
    & & \ddots & \\
    m_{d1} & m_{d2} & & m_{dd} \\
\end{bmatrix} 
\\ &=
\sum^d_{i=1} \sigma_i m_{ii}
\\ & \le
\sum^d_{i=1} \sigma_i
\end{align*}
$$

Given this inequality, in order to maximize $\frac{1}{2} \sum^n_{i=1} 2\bold{q}^\text{T}_i \bold{R} \bold{q}'_i$ (this is same as maximizing the trace result), there should be $m_{ii}=1$. Note that $\bold{M}$ itself is orthogonal rendering $\sum^d_{i=1} m_{ij}^2 = 1$, so that other terms must be zero such as $m^2_{ij}=0,\forall i \ne j$, hence, $\bold{M}$ is exactly the identity matrix $\bold{M}=\bold{I}$.

The optimal $\bold{R}^*$ can be derived by
$$
\begin{align*}
&
\bold{I} = 
\bold{M}=\bold{V}^\text{T} \bold{R}^* \bold{U}
\\ \Rightarrow \quad &
\bold{V} = 
\bold{R}^* \bold{U}
\\ \Rightarrow \quad &
\bold{R}^* = 
\bold{V}\bold{U}^\text{T}
\end{align*}
$$

With derived optimal rotation matrix $\bold{R}^*$, the optimal translation $\bold{t}^*$ can be computed as
$$
\bold{t}^* = \bold{p} - \bold{R}^*\bold{p}'
$$

## Using Non-Linear Optimization

Alternatively, the least square problem can be solved by Lie algebra. Define a six-dimensional vector $\bold{\xi}$ representing rotation and translation, and $\bold{\xi}^\wedge$ is the skew-symmetric matrix representation of $\bold{\xi}$ .
$$
\bold{\xi}=[\bold{R} | \bold{t}]
, \quad 
\bold{\xi}^\wedge =
\begin{bmatrix}
    \bold{\phi}^\wedge & \bold{\rho} \\
    \bold{0} & \bold{0}
\end{bmatrix} 
$$
where $\bold{\phi}^\wedge$ is the skew-symmetric matrix representation of rotation vector $\bold{\phi}$.

$$
arg \space \underset{\bold{\xi}}{min} \space
\frac{1}{2} \sum^n_{i=1} 
\big|\big|
    \bold{p}_i - e^{\bold{\xi}^\wedge} \bold{p}'_i
\big|\big|^2_2
$$

By introducing a trivial perturbation $\Delta \bold{\xi}$, 
$$
\begin{align*}
\frac{\partial \bold{e}_i}{\partial \bold{\Delta \bold{\xi}}}
&=
\underset{\Delta \bold{\xi} \rightarrow 0}{lim}
    \frac
    {e^{\Delta \bold{\xi}^{\wedge}}e^{ \bold{\xi}^{\wedge}}\bold{p}'_i-e^{ \bold{\xi}^{\wedge}}\bold{p}'_i}
    {\Delta \bold{\xi}}
\\ &\approx
(e^{ \bold{\xi}^{\wedge}}\bold{p}'_i)^\odot
\\ &=
\begin{bmatrix}
    \bold{I} & \bold{R}\bold{p}'_i+\bold{t} \\
    \bold{0} & \bold{0} 
\end{bmatrix}
\end{align*}
$$