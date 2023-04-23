# Similar Transform Group $Sim(3)$ and $sim(3)$

$Sim(3)$ adds scale information for monocular vision.

Define a $3$-d point $\bold{p}$ and its transformation result $\bold{p}'$
$$
\begin{align*}
    \bold{p}' &= \begin{bmatrix}
    s\bold{R} & \bold{t} \\
    \bold{0} & 1
    \end{bmatrix}
    \bold{p}
    \\ &=
    \begin{bmatrix}
    s\bold{R} & \bold{t} \\
    \bold{0} & 1
    \end{bmatrix}
    \begin{bmatrix}
        \bold{p} \\
        {1}
    \end{bmatrix}
    \\ &=
    \begin{bmatrix}
        s\bold{R}\bold{p} + \bold{t} \\
        {1}
    \end{bmatrix}
    \\ & :=
    s\bold{R}\bold{p} + \bold{t}
\end{align*}
$$

Here give the definition to $Sim(3)$ and $sim(3)$. $\bold{\zeta}$ is a 7-dimensional
vector that has the same elements as $se(3)$ plus one scaling factor $\sigma$.
$$
\begin{align*}
Sim(3) &= \bigg\{
    \bold{S} = \begin{bmatrix}
        s\bold{R} & \bold{t} \\
        \bold{0} & 1
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
\bigg\}
\\
sim(3) &= \bigg\{
    \bold{\zeta} = \begin{bmatrix}
        \bold{\rho} \\
        \bold{\phi} \\
        \sigma
    \end{bmatrix}
    \in \mathbb{R}^{7}
    , \quad
    \bold{\zeta}^\wedge =
    \begin{bmatrix}
        \sigma \bold{I}+\bold{\phi}^\wedge & \bold{\rho} \\
        \bold{0} & \bold{0}
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
\bigg\}
\end{align*}
$$

## Inverse of $Sim(3)$

$$
Sim^{-1}(3) = \bigg\{
    \bold{S}^{-1} = \begin{bmatrix}
        \frac{1}{s}\bold{R}^{\top} & - \Big(\frac{1}{s}\bold{R}^{\top}\Big) \bold{t} \\
        \bold{0} & 1
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
\bigg\}
$$

Should be easy to see that
$$
\begin{align*}
\bold{S}^{-1}
\begin{bmatrix}
    \bold{p}' \\
    {1}
\end{bmatrix}
&=
\begin{bmatrix}
    \frac{1}{s}\bold{R}^{\top} & - \Big(\frac{1}{s}\bold{R}^{\top}\Big) \bold{t} \\
    \bold{0} & 1
\end{bmatrix}
\begin{bmatrix}
    \bold{p}' \\
    {1}
\end{bmatrix}
\\ &=
\begin{bmatrix}
    \frac{1}{s}\bold{R}^{\top} & - \Big(\frac{1}{s}\bold{R}^{\top}\Big)\bold{t} \\
    \bold{0} & 1
\end{bmatrix}
\begin{bmatrix}
    s\bold{R}\bold{p} + \bold{t} \\
    {1}
\end{bmatrix}
\\ &=
\begin{bmatrix}
    \bold{p} + \frac{1}{s}\bold{R}^{\top}\bold{t} - \Big(\frac{1}{s}\bold{R}^{\top}\Big) \bold{t} \\
    {1}
\end{bmatrix}
\\ &=
\begin{bmatrix}
    \bold{p} \\
    {1}
\end{bmatrix}
\end{align*}
$$

## Solve $Sim(3)$ by Closed-form Solution of Absolute Orientation Using Unit Quaternions

Take three map points from the left hand side camera $\{\bold{r}_{l,1}, \bold{r}_{l,2}, \bold{r}_{l,3}\}$; 
three map points from the right hand side camera $\{\bold{r}_{r,1}, \bold{r}_{r,2}, \bold{r}_{r,3}\}$; 

Take $\bold{r}_{l,1}$ as the origin for the left hand side coordinate, then define the estimates for three dimensions:
* $\hat{\bold{x}}_l = {\bold{x}_l}/{||\bold{x}_l||},\qquad \bold{x}_l = \bold{r}_{l,2}-\bold{r}_{l,1}$
* $\hat{\bold{y}}_l = {\bold{y}_l}/{||\bold{y}_l||},\qquad \bold{y}_l = (\bold{r}_{l,3}-\bold{r}_{l,1}) - \big( (\bold{r}_{l,3}-\bold{r}_{l,1}) \cdot \hat{\bold{x}}_l \big)\hat{\bold{x}}_l$
* $\hat{\bold{z}}_l = {\bold{z}_l}/{||\bold{z}_l||},\qquad \bold{z}_l = \hat{\bold{x}}_l \times \hat{\bold{y}}_l$
where $\big( (\bold{r}_{l,3}-\bold{r}_{l,1}) \cdot \hat{\bold{x}}_l \big)\hat{\bold{x}}_l$ is the projection on the $\hat{\bold{x}}_l$ axis.

Set $M_l = [\hat{\bold{x}}_l, \hat{\bold{y}}_l, \hat{\bold{z}}_l]$ and $M_r = [\hat{\bold{x}}_r, \hat{\bold{y}}_r, \hat{\bold{z}}_r]$

<div style="display: flex; justify-content: center;">
      <img src="imgs/sim3_computation.png" width="20%" height="20%" alt="sim3_computation" />
</div>
</br>

For any vector on the left hand side coordinate $\bold{r}_l$, assume a transform such that $\bold{r}_r = sR(\bold{r}_l) + \bold{t}$.
The algorithm below attempts to find the optimal $s^*$, $R^*$ and $\bold{t}^*$ given the corresponding points $\bold{r}_l$ and $\bold{r}_r$

* **Find the optimal translation $\bold{t}^*$**

For any vector $\bold{r}_{l,i}$, attempt to find $\hat{\bold{r}}_{r,i} = s R( \bold{r}_{l,i}) + \bold{t}$, where $\bold{t}$ is the translation offset from the left to right coordinate system.
Here $s$ is a scale factor to rotation matrix $R( \bold{r}_{l,i})$ that has $\big|\big| R(\bold{r}_{l,i}) \big|\big|^2 = \big|\big| \bold{r}_{l,i} \big|\big|^2$ preserving the length during rotation operation ($\big|\big| \bold{r}_{l,i} \big|\big|^2=\bold{r}_{l,i} \cdot \bold{r}_{l,i}$).

The residual of the least squared problem to find the optimal $\bold{t}^*$ is defined as below.
$$
\begin{align*}
\bold{t}^* = \argmin_{\bold{t}} \bold{e}_i 
&= 
\bold{r}_{r,i} - \hat{\bold{r}}_{r,i} 
\\ &= 
\bold{r}_{r,i} - s R( \bold{r}_{l,i}) - \bold{t}    
\end{align*}
$$

Now, compute centroids served as offsets.
$$
\overline{\bold{r}}_l = \frac{1}{n} \sum_{i=1}^n \bold{r}_{l,i}
\qquad
\overline{\bold{r}}_r = \frac{1}{n} \sum_{i=1}^n \bold{r}_{r,i}
$$

For any vector $\bold{r}_{l,i}$ or $\bold{r}_{r,i}$, move/offset their coordinates from the origin reference $\bold{r}_{l,1}$ and $\bold{r}_{r,1}$ to the above computed centroid, denote the new origin's vectors as $\bold{r}'_{l,i}$ and $\bold{r}'_{r,i}$.
$$
\bold{r}'_{l,i} = \bold{r}_{l,i} - \overline{\bold{r}}_l
\qquad
\bold{r}'_{r,i} = \bold{r}_{r,i} - \overline{\bold{r}}_r
$$

Apparently, the new centroid reference's vectors' sums should be zeros.
$$
\bold{r}'_{l,o} = \sum_{i=1}^n \bold{r}'_{l,i} = [0 \quad 0 \quad 0]^{\top}
\qquad
\bold{r}'_{r,o} = \sum_{i=1}^n \bold{r}'_{r,i} = [0 \quad 0 \quad 0]^{\top}
$$

Rewrite the residual,
$$
\bold{e}_i = \bold{r}_{r,i}' - s R( \bold{r}_{l,i}') - \bold{t}'
$$
where
$$
\bold{t}' =  \bold{t} - \overline{\bold{r}}_r + sR(\overline{\bold{r}}_l)
$$

So that the least squared problem becomes finding the optimal $\bold{t}'$
$$
\begin{align*}
\min_{\bold{t}'} \sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2 
&= 
\sum_{i=1}^n \big|\big| \bold{r}_{r,i}' - s R( \bold{r}_{l,i}') - \bold{t}' \big|\big|^2
\\ &=
\sum_{i=1}^n \big|\big| \bold{r}_{r,i}' - s R( \bold{r}_{l,i}') \big|\big|^2
- \underbrace{2 \bold{t}' \cdot \sum_{i=1}^n \Big( \bold{r}_{r,i}' - s R( \bold{r}_{l,i}') \Big)}_{=\bold{0}}
+ n \big|\big| \bold{t}' \big|\big|^2
\end{align*}
$$

The sum in the middle of this expression is zero since the measurements are referred to the centroid. 

The first term does not depend on $\bold{t}'$, and the last term cannot be negative. 
So that $\sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2$ reaches its minimum when $\bold{t}'=\bold{0}$.

Rewrite $\bold{t}' = \bold{0} = \bold{t} - \overline{\bold{r}}_r + sR(\overline{\bold{r}}_l)$, so that the optimal translation $\bold{t}^*$ in $Sim(3)$ is just the difference between $\overline{\bold{r}}_r$ and scaled rotation $sR(\overline{\bold{r}}_l)$.
In other words, if $sR(\overline{\bold{r}}_l)$ is known, the $\bold{t}^*$ can easily computed.
$$
\bold{t}^* =  \overline{\bold{r}}_r - sR(\overline{\bold{r}}_l)
$$

Having said $\bold{t}' = \bold{0}$, the error can be expressed as
$$
\sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2 
=
\sum_{i=1}^n \big|\big| \bold{r}_{r,i}' - s R( \bold{r}_{l,i}') \big|\big|^2
$$

* **Find the optimal scale $s^*$**

Expand the error term

$$
\begin{align*}
&&
\sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2 
&=
\sum_{i=1}^n \big|\big| \bold{r}_{r,i}' - s R( \bold{r}_{l,i}') \big|\big|^2
\\ && &=
\sum_{i=1}^n \big|\big| \bold{r}_{r,i}' \big|\big|^2 
-2s \sum_{i=1}^n \Big( \bold{r}_{r,i}' \cdot R( \bold{r}_{l,i}')  \Big)
+ \sum_{i=1}^n \underbrace{ \big|\big| R( \bold{r}_{l,i}') \big|\big|^2}_{
    \begin{matrix}
        =\big|\big| \bold{r}_{l,i}' \big|\big|^2  \\
        \text{ for they have} \\
        \text{the same length}
    \end{matrix}
}
\\ \text{Just rewrite the notations}
&& &=
S_r - 2sD + s^2 S_l
\\ && &=
\underbrace{\Big( s\sqrt{S_l} - \frac{S}{\sqrt{S_l}} \Big)^2}_{\ge 0}
+ \frac{S_r S_l - D^2}{S_l}
\end{align*}
$$

The above quadratic term can have the optimal $s^*=\frac{D}{S_l}$ (derived by $\Big( s\sqrt{S_l} - \frac{S}{\sqrt{S_l}} \Big)^2=0$ ):
$$
s^*=\frac{D}{S_l}
=\frac{\sum_{i=1}^n \Big( \bold{r}_{r,i}' \cdot R( \bold{r}_{l,i}')  \Big)}
{\sum_{i=1}^n \big|\big| R( \bold{r}_{l,i}') \big|\big|^2}
$$ 

Now, consider the inverse transform from the right coordinate system to the left one:
$$
s^{-1}=\frac{D^{-1}}{S_l}
=\frac{\sum_{i=1}^n \Big( \bold{r}_{l,i}' \cdot R( \bold{r}_{r,i}')  \Big)}
{\sum_{i=1}^n \big|\big| R( \bold{r}_{r,i}') \big|\big|^2}
\ne \frac{1}{s} \text{ likely for the most of the time}
$$
where $\big|\big| R( \bold{r}_{r,i}') \big|\big|^2=\big|\big| \bold{r}_{l,i}' \big|\big|^2$ is constant.

This expression $s^{*\space -1} \ne \frac{1}{s}$ means that, the error computed with respect to scale $s$ according to transform from the left's to the right's $\bold{e}_{i, l \rightarrow r}=\bold{r}_{r,i}' - s R( \bold{r}_{l,i}')$ does not have the inverse scale $\frac{1}{s}$ when transformed from the right's to the left's.
In other words, the inverse transform error $\bold{e}_{i, r \rightarrow l}$ would see asymmetrical $s^{-1}$.

Unless the left-to-right transform has much more precision than the right-to-left's that $\bold{e}_{i, l \rightarrow r}=\bold{r}_{r,i}' - s R( \bold{r}_{l,i}')$ becomes accurate, otherwise, to formulate the error with respect to the scale $s$, it is better use the below symmetrical error that balances between the left-to-right and right-to-left transforms:
$$
\bold{e}_i = 
\frac{1}{\sqrt{s}}\bold{r}'_{r,i} - \sqrt{s} R (\bold{r}_{l,i})
$$

The least squared problem becomes
$$
\begin{align*}
\sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2 
&=
\frac{1}{s}S_r - 2D + s S_l
\\ &= 
\underbrace{\Big( \sqrt{s} {S_l} - \frac{1}{\sqrt{s}} S_r \Big)^2}_{\ge 0}
+ 2(S_l S_r -D)
\end{align*}
$$

The optimal $s^*=\frac{S_r}{S_l}$ can be found when $\Big( \sqrt{s} {S_l} - \frac{1}{\sqrt{s}} S_r \Big)^2=0$:
$$
s^* = \sqrt{
    \frac{ \sum_{i=1}^n \big|\big| {\bold{r}'_{r,i}} \big|\big|^2 }
    { \sum_{i=1}^n \big|\big| {\bold{r}'_{l,i}} \big|\big|^2 }
}
$$
which has a great form where rotation $R$ is removed, that the optimal scale computation only concerns the vectors/map points ${\bold{r}'_{l}}$ and ${\bold{r}'_{r}}$ in the left and right coordinate systems.

The error $\sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2 = \underbrace{\Big( \sqrt{s} {S_l} - \frac{1}{\sqrt{s}} S_r \Big)^2}_{\ge 0} + 2(S_l S_r -D)$ reaches its minimum when $D=\sum_{i=1}^n \Big( \bold{r}_{r,i}' \cdot R( \bold{r}_{l,i}')  \Big)$ grows to maximum.


* **Find the optimal rotation $R^*$**

Denote $\mathring{\bold{r}}$ as the quaternion form of $\bold{r}$:
$$
\mathring{\bold{r}} = 
r_0 + \overrightarrow{i}r_x + \overrightarrow{j}r_y + \overrightarrow{k}r_z
$$

Express $R$ in quaternion form: $\bold{r}$ rotation by quaternion $\mathring{\bold{q}}$ can be expressed as
$$
\mathring{\bold{r}}' = \mathring{\bold{q}} \mathring{\bold{r}} \mathring{\bold{q}}^{\dagger}
$$
where the rotation is defined as rotating an angle of $\theta$ about the axis defined by the unit vector $\bold{u}$ such that $\mathring{\bold{q}} = \cos \frac{\theta}{2} + \sin\frac{\theta}{2} \big( \overrightarrow{i}u_x + \overrightarrow{j}u_y + \overrightarrow{k}u_z \big)$.
Here $\mathring{\bold{q}}^{\dagger}$ is the normalization term.

Then, 
$$
M
= \sum_{i=1}^{n} \bold{r}'_{l,i} \bold{r'}_{l,i}^{\top}
= \begin{bmatrix}
    S_{xx} & S_{xy} & S_{xz} \\
    S_{yx} & S_{yy} & S_{yz} \\
    S_{zx} & S_{zy} & S_{zz} \\
\end{bmatrix}
$$
where, for example, $S_{xx}=\sum_{i=1}^{n} x'_{l,i} x'_{r,i}, S_{xy}=\sum_{i=1}^{n} x'_{l,i} y'_{r,i}$.

Recall that $D=\sum_{i=1}^n \Big( \bold{r}_{r,i}' \cdot R( \bold{r}_{l,i}')  \Big)$  needs to grow to maximum for 
$\sum_{i=1}^n \big|\big| \bold{e}_i \big|\big|^2 = \underbrace{\Big( \sqrt{s} {S_l} - \frac{1}{\sqrt{s}} S_r \Big)^2}_{\ge 0} + 2(S_l S_r -D)$ reaching its minimum.
Rewrite $D$'s elements to that $\Big( \mathring{\bold{q}} \mathring{\bold{r}}_{l,i}' \mathring{\bold{q}}^{\dagger} \Big) \cdot \mathring{\bold{r}}_{r,i}' =\Big( \mathring{\bold{q}}\bold{r}_{l,i}' \Big) \cdot \Big(  \mathring{\bold{r}}_{r,i}' \mathring{\bold{q}} \Big)$.

Take $\bold{r}_{l,i}' \rightarrow \mathring{\bold{r'}}_{l,i}$, then by quaternion multiplication, there is
$$
\mathring{\bold{q}} \mathring{\bold{r}}_{l,i}' = 
\begin{bmatrix}
    0 & -x'_{l,i} & -y'_{l,i} & -z'_{l,i} \\
    x'_{l,i} & 0 & z'_{l,i} & -y'_{l,i} \\
    y'_{l,i} & -z'_{l,i} & 0 & x'_{l,i} \\
    z'_{l,i} & y'_{l,i} & -x'_{l,i} & 0 \\
\end{bmatrix}
\mathring{\bold{q}}
=\overline{\mathcal{R}}_{l,i} \mathring{\bold{q}}
$$

Similarly, there is $\mathring{\bold{r}}_{r,i}' \mathring{\bold{q}} = \mathcal{R}_{r,i} \mathring{\bold{q}}$.

So that, $D$ can be expressed as
$$
\begin{align*}
D &=
\sum_{i=1}^{n} \Big( \mathring{\bold{q}}\bold{r}_{r,i}' \Big) \cdot \Big( \mathring{\bold{q}} \mathring{\bold{r}}_{l,i}' \Big)
\\ &=
\sum_{i=1}^{n} \Big( \overline{\mathcal{R}}_{l,i} \mathring{\bold{q}} \Big) \cdot \Big( {\mathcal{R}}_{r,i} \mathring{\bold{q}}  \Big)
\\ &=
\sum_{i=1}^{n} \mathring{\bold{q}}^{\top} 
\underbrace{\overline{\mathcal{R}}_{l,i}^{\top} {\mathcal{R}}_{r,i} }_{=N_i}
\mathring{\bold{q}}
\\ &=
\mathring{\bold{q}}^{\top} \Big( \sum_{i=1}^{n} N_i \Big) \mathring{\bold{q}}
\\ &=
\mathring{\bold{q}}^{\top} N \mathring{\bold{q}}
\end{align*}
$$

The $N$ can be expressed as
$$
N = \begin{bmatrix}
    S_{xx}+S_{yy}+S_{zz} & S_{yz}-S{zy} & S_{zx}-S{xz} & S_{xy}-S{yx} \\
    S_{yz}-S{zy} & S_{xx}-S_{yy}-S_{zz} & S_{xy}+S{yx} & S_{zx}+S{xz} \\
    S_{zx}-S{xz} & S_{xy}+S{yx} & -S_{xx}+S_{yy}-S_{zz} & S_{yz}+S{zy} \\
    S_{xy}-S{yx} & S_{zx}+S{xz} & S_{yz}+S{zy} & -S_{xx}-S_{yy}+S_{zz} \\
\end{bmatrix}
$$

Here $N$ is a real symmetric having $10$ independent elements serving the sums of the $9$ elements of $M$.
The sum of the diagonal of $N$ is zero.
In other words, the trace $tr(N)=0$ takes care of the $10$-th degree of freedom.

To maximize $\mathring{\bold{q}}^{\top} N \mathring{\bold{q}}$ by adjusting rotation $\mathring{\bold{q}}$, here computes $\text{det}(N-\lambda I)=0$, where the largest eigenvalue $\lambda_{max}$ corresponding eigenvector $\bold{v}$ is the optimal quaternion $\mathring{\bold{q}}^*$.

Given $\text{det}(N-\lambda I)=0$, compute all four eigenvalues and eigenvectors $N \bold{v}_i = \lambda_i \bold{v}_i$ for $i \in \{ 1,2,3,4 \}$.
Then, an arbitrary quaternion $\mathring{\bold{q}}$ can be written as a linear combination in the form
$$
\mathring{\bold{q}} = 
\alpha_1 \bold{v}_1 + \alpha_2 \bold{v}_2 + \alpha_3 \bold{v}_3 + \alpha_4 \bold{v}_4
$$

Since the eigenvectors are orthogonal, and for unit quaternion the norm should be $1$, there is
$$
\mathring{\bold{q}} \cdot \mathring{\bold{q}} =
\alpha_1^2 + \alpha_2^2 + \alpha_3^2 + \alpha_4^2 = 1
$$

Then,
$$
N \mathring{\bold{q}} =
\alpha_1 \lambda_1 \bold{v}_1 + \alpha_2 \lambda_2 \bold{v}_2 + \alpha_3 \lambda_3 \bold{v}_3 + \alpha_4 \lambda_4 \bold{v}_4
$$

and
$$
\mathring{\bold{q}}^{\top} N \mathring{\bold{q}} =
\mathring{\bold{q}}^{\top} \cdot \big( N \mathring{\bold{q}} \big) =
\alpha_1^2 \lambda_1 + \alpha_2^2 \lambda_2 + \alpha_3^2 \lambda_3 + \alpha_4^2 \lambda_4
$$

Sort the eigenvalues so that $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge \lambda_4$.

$\mathring{\bold{q}}^{\top} N \mathring{\bold{q}}$ reaches its maximum when $\alpha_1=1$ and $\alpha_2=\alpha_3=\alpha_4=0$.
$$
\mathring{\bold{q}}^{\top} N \mathring{\bold{q}} \le
\alpha_1^2 \lambda_1 + \alpha_2^2 \lambda_1 + \alpha_3^2 \lambda_1 + \alpha_4^2 \lambda_1
= \lambda_1
$$

This proves that when $\mathring{\bold{q}}=\bold{v}_1$ the error term $\mathring{\bold{q}}^{\top} N \mathring{\bold{q}}$ can reach its maximum.