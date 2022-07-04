# Newton's Method

## Newton's method in finding roots

Given a first-order differentiable function $f$ whose derivative is denoted as $f'$, and an initial guess $x_0$ for a root of $f$, a more accurate root can be approximated by iteration:
$$
x_{k+1}=x_{k}-\frac{f(x_k)}{f'(x_k)}
$$

## Newton's method in optimization

Newton's method attempts to find critical points (such as minima, maxima or saddle points) of a twice-differentiable function $f$, aka solutions to $f'(x)=0$.

Assume that we want to do
$$
arg \space \underset{x \in \mathbb{R}}{min} \space f(x)
$$

Newton's method attempts to solve this problem by constructing a sequence $\{x_k\}$ from an initial guess $x_0 \in \mathbb{R}$ that converges via the second-order of Talor expansion of $f$ around $x_k$:
$$
f(x_k+t)\approx f(x_k) + f'(x_k)t + \frac{1}{2}f''(x_k)t^2
$$ 

To approach to minima, the first-order derivative should be zero.
$$
0
=
\frac{d}{dt} (f(x_k) + f'(x_k)t + f''(x_k)t^2)
=
f'(x_k)+f''(x_k)t
$$

Hence,
$$
t=-\frac{f'(x_k)}{f''(x_k)}
$$

So that the iteration approaching to minima is
$$
x_{k+1}=x_k+t=x_k-\frac{f'(x_k)}{f''(x_k)}
$$

### Higher Dimension

For higher dimension $d > 1$, there is
$$
f''(x)=\triangledown^2 f(x) = H_{f}(x) \in \mathbb{R}^{d \times d}
$$
where $H$ is a Hessian matrix.

So that
$$
x_{k+1}=x_k-H_{f}^{-1}(x) f'(x_k)
$$

### Geometry

Geometrically speaking, $\frac{f'(x_k)}{f''(x_k)}$ is called *curvature* that converges fast when far from minima, slow when near to minima.

for example, given $f(x)=x^2$ with an analytic minima solution $f(0)=0$

* when it approaches to minima such as $f(0.01)=0.0001$, there is a small iterative convergence step $\frac{f'(0.01)}{f''(0.01)}=\frac{0.02}{2}=0.01$

* when it is far from minima such as $f(100)=10000$, there is a big iterative convergence step $\frac{f'(100)}{f''(100)}=\frac{200}{2}=100$