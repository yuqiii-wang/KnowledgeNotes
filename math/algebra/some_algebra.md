# Some Miscellaneous Algebra

## Set

Terminologies of a set are shown as below.

<div style="display: flex; justify-content: center;">
      <img src="imgs/set_terminologies.png" width="40%" height="20%" alt="set_terminologies" />
</div>
</br>

### Use the concept of set to prove $0.\dot{9}=1$

If $0.\dot{9} \ne 1$, there must be a number $n$ in the real axis $\mathbb{R}$ where $n \in (0.\dot{9}, 1)$, however, such $n$ does not exist.
Hence it can be said that $0.\dot{9}=1$.

### Hilbert Hotel Paradox

Hilbert Hotel Paradox is to explain infinite sets have counterintuitive properties through crafted mapping.
In particular, for an infinite set, it can have mappings to multiple/infinite infinite sets.

P.S., an infinite set is a set composed of infinite number of elements.

#### Room Setup Scenario

Hilbert's Hotel is an imaginary hotel with an infinite number of rooms, each numbered 1, 2, 3, 4, and so on.
Importantly, all the rooms are occupied—each room has a guest.

#### Through a new mapping, there will always be empty rooms

* For a new coming guest, hotel manager asks every guest move from $n$-th room to the $n+1$-th room (the mapping rule is $n+1$)
* If a bus arrives with an infinite number of guests, hotel manager ask every guests move to the $2n$-th hotel (the mapping rule is $2n$)
* For infinite number of bus arrivals with infinite number of guests, there still exists a mapping to accommodate all guests

-> Inputs:

* There are an infinite number of buses (Bus 1, Bus 2, Bus 3, …).
* Each bus has an infinite number of passengers (Seat 1, Seat 2, Seat 3, …).

-> Constraints:

* The hotel already has an infinite number of rooms, but all occupied (Room 1, Room 2, Room 3, …).

-> Outputs:

* A new mapping that maps the pair $(b, s)$ for (bus, seat) to $(r)$ for (room).

-> Solution by *Cantor Pairing Function* $CP(b, s)$

$$
r = CP(b, s) = \frac{(b+s-1)(b+s)}{2}+s
$$

The Cantor Pairing Function ensures that any countably infinite set (like the set of all possible pairs of bus and seat numbers) can be mapped to another countably infinite set (the set of room numbers) without missing or duplicating any guests.

### Find a mapping from $(0,1)$ to $[0,1]$

This question asks to establish a mapping that every element in $(0,1)$ has one or more mappings to elements in $[0,1]$.
There can be more than one mappings.

## Gamma Function

Gamma function is the generalization of factorial that is only applied to positive integers.

Recall the typical definition of factorial and it can be folded as recursive.

$$
\begin{align*}
    n! &= n \cdot (n-1) \cdot (n-2) \cdot ... \cdot 1 \\
    n! &= n \cdot (n-1)!
\end{align*}
$$

Generalize this concept to

$$
\Gamma(x+1)=x\cdot\Gamma(x)\qquad \mathcal{R}(x)>0
$$

Use exponential form to approximate this generalization form

$$
\Gamma(x)=\int_0^{\infty}t^{x-1}e^{-t} dt
$$

## Cauchy-Schwarz Inequality 柯西(-施瓦茨)不等式

In a naive algebra term, Cauchy-Schwarz inequality states that (equality holds when $ad=bc$)

$$
(a^2+b^2)(c^2+d^2)\ge (ac+bd)^2
$$

Proof: expand the expression, there are

$$
\begin{align*}
    \text{LHS}&=a^2c^2+a^2d^2+b^2c^2+b^2d^2 \\
    \text{RHS}&=a^2c^2+b^2d^2+2abcd
\end{align*}
$$

By $\text{LHS}-\text{RHS}$, there is

$$
a^2d^2+b^2c^2-2abcd=(ad-bc)^2\ge 0
$$

### Cauchy-Schwarz Inequality in Continuous Integration Form

In continuous integration, there is

$$
\int^b_a f^2(x) dx \cdot \int^b_a g^2(x) dx \ge
\bigg(\int^b_a f(x)g(x) dx\bigg)^2
$$

### Cauchy-Schwarz Inequality to Vector Space

#### Inner Product Space

The inner product space of two vectors $\bold{u}$ and $\bold{v}$ is defined as an operator that turns a space into a scalar, often denoted with angle brackets such as in $\langle \bold{u}, \bold{v} \rangle$.

The most typical example is Euclidean vector space that a function $\langle .\space, .\space \rangle : \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}$ is an inner product on $\mathbb{R}^n$.

$$
\langle \bold{u}, \bold{v} \rangle =
\Bigg\langle \begin{bmatrix}
    u_1 \\ u_2 \\ \vdots \\ u_n
\end{bmatrix},
\begin{bmatrix}
    v_1 \\ v_2 \\ \vdots \\ v_n
\end{bmatrix} \Bigg\rangle =
\bold{u}^{\top} \bold{v} =
\sum^n_{i=1} u_1 v_1 + u_2 v_2 + ... u_n v_n
$$

The definition (Euclidean vector space) holds if and only if there exists a symmetric positive-definite matrix $M$ such that (set $n=2$ as an example for illustration).

$$
\langle \bold{u}, \bold{v} \rangle :=
\bold{u}^{\top} M \bold{v} =
\begin{bmatrix}
    u_1, u_2
\end{bmatrix} \begin{bmatrix}
    m_a, m_b \\ m_b, m_d
\end{bmatrix} \begin{bmatrix}
    v_1 \\ v_2
\end{bmatrix}
$$

where $m_a>0$ and $m_d>0$ that satisfy $m_a m_d > m_b^2$ (symmetric positive-definite).
This condition says $\text{det}(M)=m_a m_d - m_b^2 > 0$ that keeps the transform $\bold{u}^{\top} M \bold{v}$ always positive.

For example, if $M$ is an identity matrix, $\langle \bold{u}, \bold{v} \rangle$ is simply a dot product.

#### Cauchy-Schwarz Inequality Definition

For all vectors $\bold{u}$ and $\bold{v}$ of an inner product space, there is

$$
|\langle \bold{u}, \bold{v} \rangle|^2 \le
\langle \bold{u}, \bold{u} \rangle \cdot \langle \bold{v}, \bold{v} \rangle
$$

Given the definition $||\bold{u}||:=\sqrt{\langle \bold{u}, \bold{u} \rangle}$ and $||\bold{v}||:=\sqrt{\langle \bold{v}, \bold{v} \rangle}$, here derives

$$
|\langle \bold{u}, \bold{v} \rangle| \le
||\bold{u}||\space||\bold{v}||
$$

where the equality is established when $\bold{u}$ and $\bold{v}$ are linearly independent.

#### Cauchy-Schwarz Inequality Geometry Explanation

The inner product $\langle \bold{u}, \bold{v} \rangle$ can be thought of as the multiplication of the length of a vector $\bold{u}$'s projection on another vector $\bold{v}$'s length.
The projection over $\cos\theta_{\bold{u}\bold{v}} \le 1$ shows the inequality.

$$
\begin{align*}
&& \cos\theta_{\bold{u}\bold{v}} &=
\frac{\langle \bold{u}, \bold{v} \rangle}{||\bold{u}||\space||\bold{v}||} \\
\Rightarrow && \langle \bold{u}, \bold{v} \rangle &=
||\bold{u}||\space||\bold{v}|| \cos\theta_{\bold{u}\bold{v}} \\
\Rightarrow && \langle \bold{u}, \bold{v} \rangle &\le ||\bold{u}||\space||\bold{v}||
\end{align*}
$$

### Cauchy-Schwarz Inequality to Random Variables and Expectation

Let $X$ and $Y$ be two random variables, here proves that $X \cdot Y$ achieves its minimum when $X$, $Y$ are equally distributed.

The expectation is as below.

$$
E(X \cdot Y)=\sum_{x}\sum_{y}
p(X=x,Y=y) \cdot x \cdot y
$$

where $p(X=x,Y=y)$ is the probability of the event $X=x$ and $Y=y$.

By Cauchy-Schwarz inequality, there is

$$
\big(E(X \cdot Y)\big)^2\le E(X^2) \cdot E(Y^2)
$$

Equality holds when $X$ and $Y$ are **linearly dependent**/**symmetric**/**uniformly distributed**, and, by extension, the most balanced or "equal" distributions often achieve this state.

### Math Application Tricks by Cauchy-Schwarz Inequality

#### Given $x\sqrt{1-y^2}+y\sqrt{1-x^2}=1$, find solution of $x^2+y^2$

Reference $(a^2+b^2)(c^2+d^2)\ge (ac+bd)^2$, in which equality holds when $ad=bc$
Let $x=a, y=d$ there is

$$
x\underbrace{\sqrt{1-y^2}}_{=c}+y\underbrace{\sqrt{1-x^2}}_{=b}\ge
(x^2+1-x^2)(y^2+1-y^2)=1
$$

The equality holds only when $x^2y^2=(1-x^2)(1-y^2)$, then

$$
\begin{align*}
    x^2y^2&=(1-x^2)(1-y^2) \\
    x^2y^2&=1-x^2-y^2+x^2y^2 \\
    x^2+y^2&=1
\end{align*}
$$

#### Given $x+2y=1$ and $x,y>0$, find $\min \frac{1}{x}+\frac{8}{y}$

Reference $(a^2+b^2)(c^2+d^2)\ge (ac+bd)^2$, and for $x,y>0$, let $a^2=x, b^2=2y, c^2=\frac{1}{x}, d^2=\frac{8}{y}$, then

$$
(x+2y)(\frac{1}{x}+\frac{8}{y})=\frac{1}{x}+\frac{8}{y}\ge
(1+\sqrt{2}\cdot\sqrt{8})^2=25
$$

#### For $\theta\in\mathbb{R}$, find $\max y=3\sin\theta+2\cos\theta$

Reference $(a^2+b^2)(c^2+d^2)\ge (ac+bd)^2$, let $a=3, b=2, c=\sin\theta, d=\cos\theta$, then

$$
\begin{align*}
    (2^2+3^2)(\sin^2\theta+\cos^2\theta) &\ge(3\sin\theta+2\cos\theta)^2 \\
    3\sin\theta+2\cos\theta &\le\sqrt{13}
\end{align*}
$$

As a result, $\max y=\sqrt{13}$.

## Complete homogeneous symmetric polynomial

A polynomial is complete homogeneous symmetric when satisfying the below expression 

$$
h_k(x_1,x_2,...,x_n)=
\sum_{1 \le i_1 , i_2 , ..., \le i_n}
\frac{m_1!m_2!...m_n!}{k!} 
x_{i_1} x_{i_2} ...  x_{i_n}
$$

which is basically the full combinations of variables and their powers.

### Example

For $n=3$:
$$
\begin{align*}
h_1(x_1, x_2, x_3) &= x_1 + x_2 + x_3
\\
h_2(x_1, x_2, x_3) &= x_1^2 + x_2^2 + x_3^2 + x_1 x_2 + x_2 x_3 + x_1 x_3
\\
h_3(x_1, x_2, x_3) &= x_1^3 + x_2^3 + x_3^3 +x_1^2 x_2 + x_2^2 x_3 + x_1^2 x_3 +x_2^2 x_1 + x_3^2 x_1 + x_3^2 x_2+x_1 x_2 x_3
\end{align*}
$$

## Notations

### Number Set Notations

* $\mathbb{N}$ (Natural Numbers):
  * $\mathbb{N} = \{1, 2, 3, ... \}$
* $\mathbb{Z}$ (Integers):
  * $\mathbb{Z} = \{..., -3, -2, -1, 0, 1, 2, 3, ... \}$
* $\mathbb{Q}$ (Rationals, numbers that can be expressed as a fraction of two integers, where the denominator is not zero):
  * $\mathbb{Q} = \{p/q \space|\space p \in \mathbb{Z}, q \in \mathbb{Z}, q \ne 0 \}$
  * For example, $1/2 \in \mathbb{Q}$
* $\mathbb{R}$ (Real Numbers):
  * $\mathbb{R} = \{x \space|\space x \text{ includes rational and irrational numbers} \}$
* $\mathbb{C}$ (Complex Numbers):
  * $\mathbb{C} = \{a+ib \space|\space a, b \in \mathbb{{R}}, i^2=-1 \}$
* $\mathbb{P}$ (Prime Numbers, natural numbers greater than 1 that have no positive divisors other than 1 and themselves):
  * $\mathbb{P} = \{2, 3, 5, 7, 13, 17, ... \}$

Upper symbols:

* $\mathbb{R}^+, \mathbb{N}^+$, etc., positive numbers
* $\mathbb{R}^*, \mathbb{N}^*$, etc., non-zero numbers

### Einstein Notation

Einstein notation (a.k.a Einstein summation convention or Einstein summation notation) is a notational convention that implies summation over a set of **indexed terms** in a formula, hence achieving brevity.

For example,

$$
\bold{v} = v^i e_i =
\begin{bmatrix}
    e_1 & e_2 & e_3 & ... & e_n
\end{bmatrix}
\begin{bmatrix}
    v_1 \\ v_2 \\ v_3 \\ \vdots \\ v_n
\end{bmatrix}
$$
