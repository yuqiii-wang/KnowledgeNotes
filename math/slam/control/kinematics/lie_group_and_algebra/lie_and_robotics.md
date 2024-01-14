# Lie and Robotics

## Motivation

The three-dimensional rotation
matrix $R$ constitutes the *special orthogonal group* $SO(3)$, and the transformation matrix $t$
constitutes the *special Euclidean group* $SE(3)$.
$$
\begin{align*}
SO(3) &=
\big\{
      R \in \mathbb{R}^{3 \times 3} | RR^\text{T} = I , det(R) = 1 
\big\}
\\
SE(3) &= 
\bigg\{
   T = 
    \begin{bmatrix}
        R & t \\
        0 & 1
    \end{bmatrix}
    \in \mathbb{R}^{4 \times 4}
    |
    R \in SO(3), t \in \mathbb{R}^3
\bigg\}
\end{align*}
$$

Matrix operations on $SO(3)$ and $SE(3)$ should be well defined (the operation result should be contained in $SO(3)$ and $SE(3)$ spaces). 
For example, multiplication is well defined.
$$
\begin{align*}
R_1  R_2 &\in SO(3)
\\
T_1  T_2 &\in SE(3)
\end{align*}
$$

However, there is no definition of matrix addition, such that 
$$
\begin{align*}
R_1 + R_2 &\notin SO(3)
\\
T_1 + T_2 &\notin SE(3)
\end{align*}
$$

Addition is much simpler and faster in computation than multiplication. Better to have them.

Lie group and Lie algebra come in rescue that it uses the tangent of the rotation space to represent local information. The tangent is well defined in addition.

$SO(3)$ and $SE(3)$ are Lie group, whose Lie algebra are denoted as $so(3)$ and $se(3)$.

## Lie Bracket in Rotation

Given a rotation matrix $R$ that changes with time $t$, denoted as $R(t)$, for $R(t) \in SO(3)$, there is 
$$
\begin{align*}
R(t)R(t)^T=I
\\
R(t)R(t)^{-1}=I
\end{align*}
$$

By differential, there are
$$
\dot{R(t)}R(t)^T + R(t)\dot{R(t)^T} = 0 \\
\dot{R(t)}R(t)^T = -(\dot{R(t)}R(t)^T)^T
$$
where $\dot{R(t)}R(t)^T$ is a skew-symmetric matrix.

Remember one property of skew-symmetric matrix:
the space of a skew-symmetric matrices $A_{n \times n}$ has dimensionality $\frac{1}{2} n (n - 1)$, its vector representation is $a^{\wedge}_{\frac{1}{2} n (n - 1)}$, for example, for a $3 \times 3$ matrix, there is

$$
\bold{a}^{\wedge}=
A =
\begin{bmatrix}
      0 & -a_3 & a_2 \\
      a_3 & 0 & -a_1 \\
      -a_2 & a_1    & 0
\end{bmatrix}
$$

Here uses $\wedge$ to represent a vector space of a skew-symmetric matrix.

Reversely, here defines a vector representation of a skew-symmetric matrix with $\vee$
$$
A = \bold{a}^\vee
$$

Since $\dot{R(t)}R(t)^T$ is a skew-symmetric matrix, here uses $\phi^{\wedge}$ to represent the multiplication result of $\dot{R(t)}R(t)^T$
$$
\phi^{\wedge} = \dot{R(t)}R(t)^T
$$

$$
\phi^{\wedge}R(t) = \dot{R(t)}
$$

First degree Taylor Series of $R$ at the time $t_0=0$ is shown as below.
$$
R(t) \approx R(t_0) + \dot{R(t_0)} (t-t_0)
\\= I + \phi(t_0)^{\wedge}(t)
$$

Set $t=0$,
the above approximation becomes $R(0) = I$ for $\phi(t_0)^{\wedge}(0) = 0$.

Denote $\phi(t_0)=\phi_0$ for $\phi(t_0)$ is a constant within the vicinity of $t_0$.

Now compute $R(t)$, remember $\dot{R(t)} = \phi_0^{\wedge}R(t)$ is a homogeneous linear differential equation, so that its integral result is
$$
\begin{align*}

\int \dot{R(t)} &= 
\int \phi_0^{\wedge}R(t) dt
\\
R(t) &= 
e^{\phi^{\wedge}_0t}

\end{align*}
$$

Now define $so(3)$ and $se(3)$ (intuitively speaking, they represent translation/rotation vector forms)
$$
\begin{align*}
so(3) &= 
\{
    \phi \in \mathbb{R}^3 \text{or } {\Phi}^\wedge \in \mathbb{R}^{3 \times 3}
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

whose Lie brackets are
$$
\begin{align*}
[\phi_1, \phi_2] &= 
(\Phi_1 \Phi_2 - \Phi_2 \Phi_1)^\vee
\\
[\xi_1, \xi_2] &= 
(\xi_1^\wedge \xi_2^\wedge - \xi_2^\wedge \xi_1^\wedge)^\vee
\end{align*}
$$

where $\xi^\wedge$ represents $6$-dimensional vector to $4 \times 4$-dimensional matrix transformation, and $\xi^\vee$ refers to the opposite transformation from matrix to vector.

Lie bracket describes two Lie algebras' operations that elaborate local information (the differences between two elements) of corresponding Lie groups.

## Exponential and Logarithmic Mapping

Rotation matrix $R = e^{\phi^{\wedge}}$ is a complex operation. We want to just focus on $\phi^{\wedge}$.

For example, for a 2-dimensional rotation $\bold{z} = \bold{x}e^{i\theta}$, whose Lie algebra is $i\theta$ that sees rotation at a particular point. We can just do derivative on $i\theta$ rather than the difficult $\bold{z}$, that gives approximately the same result, since the changes on tangent and direct derivative on its corresponding group are almost the same.

![2d_lie_mapping](imgs/2d_lie_mapping.png "2d_lie_mapping")

### Exponential Mapping $so(3) \mapsto SO(3)$

Define a unit vector $\bold{v}$; remember *Rodrigues’s Rotation Formula* that describes the matrix representation over a 3-dimensional angle $\theta$. Here derives the exponential mapping $so(3) \mapsto SO(3)$.
$$
\begin{align*}
R &= e^{\phi^{\wedge}} =
e^{\theta \bold{v}^{\wedge}} 
\\ &= 
\sum_{n=0}^{\infty} \frac{{(\theta \bold{v}^{\wedge}})^n}{n!}
\\ &=
cos \theta I + (1 - cos \theta)\bold{v} \bold{v}^\text{T} + sin\theta \bold{v}^{\wedge}
\end{align*}
$$

### Logarithmic Mapping $SO(3) \mapsto so(3)$

Conversely, there is $so(3) \mapsto SO(3)$, and here derives the logarithmic mapping via Taylor series expansion.

Remember $\phi^{\wedge} = \theta \bold{v}^{\wedge}$, and $\bold{v}^{\wedge} \bold{v}^{\wedge} = \bold{v}\bold{v}^\text{T}-I$, $\bold{v}^{\wedge}  \bold{v}^{\wedge} \bold{v}^{\wedge} = -\bold{v}^{\wedge}$
$$
\begin{align*}
    \phi &= ln(R)^\vee 
\\ &=
ln \big(
    e^{\phi^\wedge} 
\big)^\vee
\\ &=
  ln\bigg(
    \sum_{n=0}^{\infty} \frac{{(\phi^\wedge)^n}}{n!}
  \bigg)^\vee
\\ &=
ln\bigg(
    \sum_{n=0}^{\infty} \frac{{(\dot{R}R^\text{T})^n}}{n!}
  \bigg)^\vee
\\ &=
\bigg(
    \sum^{\infty}_{n=0} \frac{(-1)^{n-1}}{n}
    (R-I)^{n}
\bigg)^\vee
\end{align*}
$$

### Exponential Mapping $se(3) \mapsto SE(3)$

$$
\begin{align*}
e^{\xi^\wedge} &= 
\begin{bmatrix}
    \sum_{n=0}^{\infty} \frac{{(\phi^\wedge)^n}}{n!} 
    &
    \sum_{n=0}^{\infty} \frac{{(\phi^\wedge)^n}}{(n+1)!} \bold{\rho}
    \\
    0 & 1
\end{bmatrix}
\\ & \overset{\Delta}{=}
\begin{bmatrix}
    R
    &
    J \bold{\rho}
    \\
    0 & 1
\end{bmatrix}=
\begin{bmatrix}
    R
    &
    \bold{t}
    \\
    0 & 1
\end{bmatrix}
\\ &= T
\end{align*}
$$
where
$$
\begin{align*}
\sum_{n=0}^{\infty} \frac{{(\phi^\wedge)^n}}{(n+1)!}&=
I + \frac{1}{2!}\theta \bold{v}^\wedge + \frac{1}{3!}\theta (\bold{v}^\wedge)^2 + ...
\\ &=
\frac{1}{\theta}(\frac{\theta^2}{2!}+\frac{\theta^4}{4!}+...)\bold{v}^\wedge +
\frac{1}{\theta}(\frac{\theta^3}{3!}+\frac{\theta^5}{5!}+...)(\bold{v}^\wedge)^2 +I
\\ &=
\frac{\bold{v}^\wedge}{\theta}(1-cos\theta)+
\frac{\theta-sin\theta}{\theta}(\bold{v}\bold{v}^\text{T}-I)+I
\\&=
\frac{sin\theta}{\theta} I + 
(1-\frac{sin \theta}{\theta})\bold{v}\bold{v}^\text{T} +
\frac{1-cos\theta}{\theta} \bold{v}^\wedge
\\ & \overset{\Delta}{=}
J
\end{align*}
$$

The above expression says the rotation part is just the same as $R$, and the translation part is a Jacobian times the translation vector.

### Logarithmic Mapping $SE(3) \mapsto se(3)$

Simply there is
$$
\begin{align*}
\xi^\wedge &= ln(T)
\\ &=
ln \begin{bmatrix}
    R & \bold{t} \\
    0 & 1
\end{bmatrix}
\\ &=
\begin{bmatrix}
    \phi^\wedge & \rho \\
    0 & 0
\end{bmatrix}
\end{align*}
$$

### Summary

|Lie Group||Conversion||Lie Algebra|
|-|-|-|-|-|
|$SO(3) \\ R=e^{\phi^\wedge} \in \mathbb{R}^{3 \times 3}$|$\rightarrow$|Logarithmic mapping:$\\ \theta = arccos \bigg(\frac{tr(R)-1}{2}\bigg) \\ R\bold{v}=\bold{v}$|$\rightarrow$|$so(3) \\ \phi \in \mathbb{R}^3$|
||$\leftarrow$|Exponential mapping: $e^{\theta \bold{v}^\wedge}$|$\leftarrow$||
|$SE(3) \\ T \in \mathbb{R}^{4 \times 4}$|$\rightarrow$|Logarithmic mapping:$\\ \theta = arccos \bigg(\frac{tr(R)-1}{2}\bigg) \\ R\bold{v}=\bold{v} \\ \bold{t}=J\rho$|$\rightarrow$|$se(3) \\ \xi=[\phi^\wedge \quad \rho] \in \mathbb{R}^6$|
||$\leftarrow$|Exponential mapping: $e^{\xi^\wedge}$|$\leftarrow$||


## Derivative

An important question is about rotation matrix derivative, such as given a 3-d point $\bold{p}$, here to compute
$$
\frac{\partial R\bold{p}}{\partial R}
$$

Given the definition of derivative such that
$$
\frac{\partial R}{\partial \Delta R}=
lim_{\Delta R \rightarrow 0}
\frac{ (R \oplus \Delta R) - R}{\Delta R}
$$
where $\oplus \Delta R$ denotes the increment amount.
There is neither addition nor subtraction for $R$. Lie algebra can help instantiate $\oplus$.

There are two solutions to this problem.

* Derivative Model: $R$-corresponding Lie algebra $\phi$ adds a $\Delta \phi$, then compute the change rate on the $\Delta \phi$, so that 
$$
\frac{\partial R\bold{p}}{\partial R}=
lim_{\Delta \phi \rightarrow 0} \frac{e^{(\Delta \phi + \phi)^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\Delta \phi}
$$

* Perturbation Model: directly multiplying $\Delta R$ to either the left or the right of $R$ (added a trivial perturbation), then compute the derivative on the Lie algebra of the trivial perturbation $\Delta R$ denoted as $\psi$, so that
$$
\begin{align*}
\frac{\partial R\bold{p}}{\partial R}
& \approx
lim_{ \psi \rightarrow 0} \frac{e^{\psi^{\wedge}}e^{\phi^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\psi}
\quad \text{Left Perturbation}
\\ & \approx
lim_{ \psi \rightarrow 0} \frac{e^{\phi^{\wedge}}e^{\psi^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\psi}
\quad \text{Right Perturbation}
\end{align*}
$$

In practice, we set $\Delta R = [10^{-6}\quad 10^{-6}\quad 10^{-6}]^{\wedge}$. When applied perturbation to $R$, $\frac{\partial R\bold{p}}{\partial R}$ is nearly unaffected and remained the most of the derivative.

### BCH Formula and its Approximation

Recall the *BCH Formula and its Approximation*

$$
\begin{equation*}
ln(exp(\phi^{\wedge}_1)exp(\phi^{\wedge}_2))
\approx
\begin{cases}
          J_l(\phi_2)^{-1}\phi_1 + \phi_2 \quad &\text{if } \phi_1 \text{is sufficiently small} \\
          J_r(\phi_1)^{-1}\phi_2 + \phi_1 \quad &\text{if } \phi_2 \text{is sufficiently small} \\
     \end{cases}
\end{equation*}
$$

where $J_l$ and $J_r$ are 

$$
J_l = \frac{sin\theta}{\theta} I + (1 - \frac{sin\theta}{\theta})\bold{v}\bold{v}^\text{T} + \frac{1-cos\theta}{\theta}\bold{v}^{\wedge}
$$
whose derivative is 
$$
J^{-1}_l = \frac{\theta}{2}cot\frac{\theta}{2}I + (1-\frac{\theta}{2}cot\frac{\theta}{2})\bold{v}\bold{v}^\text{T} - \frac{\theta}{2}\bold{v}^{\wedge}
$$

and for the right multiple
$$
J_r(\phi) = J_l(-\phi)
$$

### Derivative Model

Now a point $p$ is rotated by $R$, hence the new position is $Rp$. To calculate $\frac{\partial Rp}{\partial R}$:
$$\begin{align*}
\frac{\partial R\bold{p}}{\partial R}
 & =
\frac{\partial e^{\phi^{\wedge}}\bold{p}}{\partial e^{\phi}}
\\ & =
lim_{\Delta \phi \rightarrow 0} \frac{e^{(\Delta \phi + \phi)^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\Delta \phi}
\\ & =
lim_{\Delta \phi \rightarrow 0} \frac{e^{(J_l \Delta \phi)^{\wedge}}e^{\phi^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\Delta \phi}
\\ & \approx
lim_{\Delta \phi \rightarrow 0} \frac{((I+J_l \Delta \phi)^{\wedge})e^{\phi^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\Delta \phi}
\\ &=
lim_{\Delta \phi \rightarrow 0} \frac{(J_l \Delta \phi)^{\wedge} e^{\phi^{\wedge}}\bold{p}}{\Delta \phi}
\\ &=
lim_{\Delta \phi \rightarrow 0} \frac{-(e^{\phi^{\wedge}}\bold{p})^\wedge J_l \Delta \phi}{\Delta \phi}
\\ &=
-(e^{\phi^{\wedge}}\bold{p})^\wedge J_l
\\ & =
-(R\bold{p})^{\wedge}J_l
\end{align*}$$

### Perturbation Model

Apply a trivial perturbation $\Delta R$ and take partial derivative over this perturbation to avoid computing the Jacobian $J_l$:
$$\begin{align*}
\frac{\partial R \bold{p}}{\partial \Delta R}
 & =
\frac{\partial e^{\phi^{\wedge}}\bold{p}}{\partial \psi}
\\ & =
lim_{ \psi \rightarrow 0} \frac{e^{\psi^{\wedge}}e^{\phi^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{\psi}\\ & \approx
lim_{ \psi \rightarrow 0} \frac{(I+\psi^{\wedge})e^{\phi^{\wedge}}\bold{p} - e^{\phi^{\wedge}}\bold{p}}{ \psi}
\\ &=
lim_{ \psi \rightarrow 0} 
\frac{\psi^{\wedge}e^{\phi^{\wedge}}\bold{p} }{\psi}
\\ &=
lim_{ \psi \rightarrow 0} 
\frac{(e^{\phi^{\wedge}}\bold{p})^\wedge \psi }{\psi}
\\ & =
-(Rp)^{\wedge}
\end{align*}$$

### Perturbation Model Considered Both Translation and Rotation

Define $[R|T]$ as the transformation matrix to a point $\bold{p}$ and the perturbation as $\Delta \bold{\xi}$ (this time, both translation $T$'s perturbation $\Delta \bold{\rho}$ and rotation $R$'s perturbation $\Delta \bold{\phi}$ are included, so that $\Delta \bold{\xi} = [\Delta \bold{\rho}, \Delta \bold{\phi}]^\text{T}$), the derivative can be computed as the below

$$
\begin{align*}
    \frac{\partial ([R|T]\bold{p})}{\partial \Delta \bold{\xi}} &= 
    \underset{\Delta \bold{\xi} \rightarrow 0}{lim}
    \frac
    {e^{\Delta \bold{\xi}^{\wedge}}e^{ \bold{\xi}^{\wedge}}\bold{p}-e^{ \bold{\xi}^{\wedge}}\bold{p}}
    {\Delta \bold{\xi}}
    \\ &=
    \underset{\Delta \bold{\xi} \rightarrow 0}{lim}
    \frac
    {(I+{\Delta \bold{\xi}^{\wedge}})e^{ \bold{\xi}^{\wedge}}\bold{p}-e^{ \bold{\xi}^{\wedge}}\bold{p}}
    {\Delta \bold{\xi}}
    \\ &=
    \underset{\Delta \bold{\xi} \rightarrow 0}{lim}
    \frac
    {{\Delta \bold{\xi}^{\wedge}}e^{ \bold{\xi}^{\wedge}}\bold{p}}
    {\Delta \bold{\xi}}
    \\ &=
    \underset{\Delta \bold{\xi} \rightarrow 0}{lim}
    \frac
    {
        \begin{bmatrix}
            \Delta \bold{\phi} & \Delta \bold{\rho} \\
            \bold{0} & \bold{0}
        \end{bmatrix}
        \begin{bmatrix}
            R\bold{p}+T \\
            \bold{1}
        \end{bmatrix}
    }
    {\Delta \bold{\xi}}
    \\ &=
    \underset{[\Delta \bold{\rho}, \Delta \bold{\phi}]^\text{T} \rightarrow 0}{lim}
    \frac
    {
        \begin{bmatrix}
            \Delta \bold{\phi} (R\bold{p}+T) + \Delta \bold{\rho} \\
            \bold{0}
        \end{bmatrix}
    }
    {\begin{bmatrix}
        \Delta \bold{\rho} \\ 
        \Delta \bold{\phi}
    \end{bmatrix}}
    \\ &=
    \begin{bmatrix}
        \frac{\partial (\Delta \bold{\phi} (R\bold{p}+T) + \Delta \bold{\rho})}{\partial \Delta \bold{\rho}} & 
        \frac{\partial (\Delta \bold{\phi} (R\bold{p}+T) + \Delta \bold{\rho})}{\partial \Delta \bold{\phi}} \\
        \frac{\partial \bold{0}}{\partial \Delta \bold{\rho}} & 
        \frac{\partial \bold{0}}{\partial \Delta \bold{\phi}} 
    \end{bmatrix}
    \\ &=
    \begin{bmatrix}
        \bold{I} & R\bold{p}+T \\
        \bold{0} & \bold{0} 
    \end{bmatrix}
\end{align*}
$$

### Perturbation Model vs Derivative Model

The derivative model uses $e^{(\Delta \phi + \phi)^{\wedge}}=e^{(J_l \Delta \phi)^{\wedge}}e^{\phi^{\wedge}}$ to represent the increment result.
$J_l$'s computation needs Taylor expansion and is complicated.

The perturbation model adds a trivial disturbance $e^{\psi^{\wedge}}e^{\phi^{\wedge}}\approx(I+\psi^{\wedge})e^{\phi^{\wedge}}$ such as $\Delta R=[10^{-6}\quad 10^{-6}\quad 10^{-6}]^{\wedge}$ to $R$ that gives the result $\Delta R R$ to see the change of the result relative to the disturbance.
Intuitively speaking, the added disturbance has little effect on the existing $R$, so that $\Delta R R$ is little changed.