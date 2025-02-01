# T Distribution

$$
f(t)=\frac{\Gamma\big(\frac{v+1}{2}\big)}{\sqrt{\pi v}\space\Gamma\big(\frac{v}{2}\big)}\bigg(1+\frac{t^2}{v}\bigg)^{-(v+1)/2}
$$

where Gamma function is $\Gamma(v)=\int_0^{\infty}t^{v-1}e^{-t} dt$.
In particular for $v\in\mathbb{Z}^{+}$, there is $\Gamma(v)=(v-1)!$.

## T Statistic

Assume samples drawn from a normal population $X_1,X_2, ..., X_n \sim \mathcal{N}(\mu, \sigma^2)$,

* Sample Mean: $\overline{X}\sim \mathcal{N}(\mu, \frac{\sigma^2}{n})$
* Sample Variance: $s^2=\frac{1}{n-1}\sum^n_{i=1}(X_i-\overline{X})^2$
* Standard normal distribution: $Z=\frac{\overline{X}-\mu}{\sigma/\sqrt{n}}\sim\mathcal{N}(0,1)$
* Chi-squared distribution: $V=\frac{(n-1)s^2}{\sigma^2}\sim\mathcal{X}^2(v),\qquad v=n-1$

The t-Statistic is

$$
t=\frac{\overline{X}-\mu}{s/\sqrt{n}}=
\frac{Z}{\sqrt{\frac{1}{v}\frac{(n-1)s^2}{\sigma^2}}}=
\frac{Z}{\sqrt{V/v}}
$$

By intuition, t-Statistic is a standard normal distribution normalized by sample size.

## Derive T Distribution

## Cauchy Distribution (1 d.o.f)

Cauchy distribution is special Student's t-Distribution with degree of freedom $v=1$.

$$
\begin{align*}
    & f_v(t) &&= \frac{\Gamma\big(\frac{v+1}{2}\big)}{\sqrt{\pi v}\space\Gamma\big(\frac{v}{2}\big)}\bigg(1+\frac{t^2}{v}\bigg)^{-(v+1)/2} \\
    \text{set } v=1 \Rightarrow\quad & &&= \frac{1}{\sqrt{\pi}\cdot\sqrt{\pi}}\bigg(1+t^2\bigg)^{-1} \\
    &&&= \frac{1}{\pi}\bigg(1+t^2\bigg)^{-1}
\end{align*}
$$

where $\Gamma(\frac{1}{2})=\int_{-\infty}^{\infty}t^{-1/2}e^{-t}dt=\sqrt{\pi}$.

The normalization $\frac{1}{\pi}$ ensures integration result to $1$.

### Proof of $\Gamma(\frac{1}{2})=\sqrt{\pi}$

Here to prove $\Gamma(\frac{1}{2}) = \int_{-\infty}^{\infty}t^{-1/2}e^{-t}dt=\sqrt{\pi}$.

#### Transform to typical Gaussian integration form

$$
\begin{align*}
    && \Gamma(\frac{1}{2}) &= \int_{-\infty}^{\infty}t^{-1/2}e^{-t}dt \\
    \text{Substitute } t=x^2, dt=2xdx\qquad \Rightarrow && &= \int_{-\infty}^{\infty} 2(x^2)^{-1/2} e^{-x^2} xdx \\
    && &= 2\int_{-\infty}^{\infty} x\cdot x^{-1} e^{-x^2} dx \\
    && &= 2\int_{-\infty}^{\infty} e^{-x^2} dx
\end{align*}
$$

Let $I=\int_{0}^{\infty} e^{-x^2} dx$, and square the integral

$$
\begin{align*}
    I^2 &= \bigg(\int_{-\infty}^{\infty} e^{-x^2} dx\bigg)\bigg(\int_{-\infty}^{\infty} e^{-y^2} dy\bigg) \\
    &= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2+y^2)}dxdy
\end{align*}
$$

#### Transform to polar coordinates

Let $x=r\sin\theta$ and $y=r\cos\theta$, then

$$
x^2+y^2=r^2(\sin^2\theta+\cos^2\theta)=r^2
$$

This fits the definition of the polar coordinates.

* $r=\sqrt{x^2+y^2}\in[0, \infty)$
* $\theta=\arctan^2(y,x)\in[0,2\pi]$

So that

$$
\begin{align*}
    I^2 &= \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2+y^2)} dxdy \\
    &= \int_{0}^{2\pi}\int_{0}^{\infty} e^{-r^2} rdrd\theta
\end{align*}
$$

#### Proof of $dxdy=rdrd\theta$

Intuitively speaking, for integration increment growth of the rectangular area $dxdy$, the corresponding increment growth in polar coordinates is the area of a sector by $dr \times rd\theta$.

The Jacobian determinant gives the growth rate

$$
J = \begin{bmatrix}
    \frac{\partial x}{dr} & \frac{\partial x}{d\theta} \\
    \frac{\partial y}{dr} & \frac{\partial y}{d\theta}
\end{bmatrix} = \begin{bmatrix}
    \sin\theta & -r\cos\theta \\
    \cos\theta & r\sin\theta \\
\end{bmatrix}
$$

The determinant $\text{det}(J)$ is

$$
\begin{align*}
    \text{det}(J) &= r\sin^2\theta+r\cos^2\theta \\
    &= r(\sin^2+\cos^2) \\
    &= r
\end{align*}
$$

#### Evaluate the Radial and Angular Integral

There are two parts in $I^2=\int_{0}^{2\pi}\int_{0}^{\infty} e^{-r^2} rdrd\theta$:

* Radial Integral $\int_{0}^{\infty} e^{-r^2} rdr$
* Angular Integral $\int_{0}^{2\pi}d\theta$

It is easy to say that

$$
\int_{0}^{2\pi}d\theta=2\pi
$$

For radial integral, let $u=r^2$, so that $du=2rdr$, then

$$
\begin{align*}
    \int_{0}^{\infty} e^{-r^2} rdr=
    \int_{0}^{\infty} e^{-u} \frac{1}{2}du=
    \frac{1}{2}
\end{align*}
$$

#### Final Integration Result

Given the above radial and angular integral results, there is

$$
I^2=\int_{0}^{2\pi}\int_{0}^{\infty} e^{-r^2} rdrd\theta=\frac{1}{2} \cdot 2\pi=\pi
$$

Therefore, having squared root the result, there is

$$
I=\int_{-\infty}^{\infty}t^{-1/2}e^{-t}dt=\sqrt{\pi}
$$
