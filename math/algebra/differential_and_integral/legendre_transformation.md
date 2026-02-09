#  Legendre Transformation

A *Legendre transform* Legendre transform converts from a function of one set of variables to another function of a conjugate set of variables. 
Both functions will have the same units. 

* Define a *conjugate* pair 

If $(x,y)$ is a conjugate pair of variables, then
$d(xy)=xdy+ydx$ relates the variation $dy$ in quantity $y$ to the variation $dx$ in quantity $x$.

Consider a function of two independent variables, call it $f(x,y)$, whose differential is

$$
df=
\bigg(\frac{\partial f}{\partial x}\bigg)_y dx
+
\bigg(\frac{\partial f}{\partial y}\bigg)_x dy
$$

Define $u \equiv \bigg(\frac{\partial f}{\partial x}\bigg)_y$ and $w \equiv \bigg(\frac{\partial f}{\partial y}\bigg)_x$, so that the above equation can be rewritten as

$$
df= u\space dx + w\space dy 
$$

Here $u$ and $x$ are called a conjugate pair of variables, and likewise $w$ and $y$.

* Define *Legendre Transformation*

Use product rule for $d(wy)$, there is

$$
d(wy) = y\space dw + w\space dy
$$

Subtract $d(wy)$ from $df$ and derive $dg$

$$
dg := df - d(wy) =
u\space dx - y\space dw
$$

Legendre transformation function is defined as $g \equiv f-wy$ and write $g(x,w)$.

Legendre transformation takes an original function $f(x,y)$ transformed to
a new function $g(x,w)$ by switching from variable $y$ to its conjugate variable $w$.

## Example: Legendre transform from the Lagrangian $L$ to the Hamiltonian $H$

A mechanical system is defined as a distance coordinate $q$ and corresponding velocity $\dot{q}$.
Then the Lagrangian is defined as the difference between the kinetic
and potential energies, $L(q, \dot{q})\equiv K-U$. 

Legendre transform is used to transform $L(q, \dot{q})$ to a new function $H(q,p)$, where $p$ is the momentum ($p=mv$, where in physics, $m$ is mass and $v$ is velocity).

* $f \equiv L$ the original function
* $x \equiv q$ the variable we are not switching
* $y \equiv \dot{q}$ the variable to be switched
* $w \equiv \big(\frac{\partial f}{\partial y}\big)_x = \big(\frac{\partial L}{\partial \dot{q}}\big)_q \equiv p$ the conjugate of the switched variable.

By the canonical momentum definition $p=mv$, consider kinetic energy definition $K=\frac{1}{2}mv^2$, where $\dot{q}=v$ as the velocity, then
$$
\big(\frac{\partial L}{\partial \dot{q}}\big)_q=
\big(\frac{\partial K}{\partial v}\big)_q= mv = p
$$

Potential energy $U$ is conservative, not a function of velocity but position $q$.

By the definition of Legendre transform, there is

$$
g \equiv f - wy =
L - p\dot{q}
$$

Add a negative sign to the above expression, so that 
$$
p\dot{q} - L = (mv) v - (\frac{1}{2}mv^2 - U)= K + U = H
$$

Hamiltonian $H$ represents the whole system energy, while $L$ says about the differences between $K$ and $U$. Legendre transform expresses this system in a different way by just taking the conjugate of the variable.