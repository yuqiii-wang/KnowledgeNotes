# Bond Valuation and Returns

## The Three Yield Curves

### Yield to Maturity (YTM) Curve (到期收益率曲线)

Yield to maturity (YTM) is the rate when bond is purchased on the secondary market, expected annualized return rate.

$$
\text{BondPrice}=\sum^n_{t=1}\frac{\text{CouponRate}}{(1+r)^t}+\frac{\text{FaceValue}}{(1+r)^n}
$$

For example, a two-year maturity, 6% coupon rate bond with a face value of 100 priced at 98 on the market, there is

$$
98=\frac{6}{1+r}+\frac{106}{(1+r)^2}, \qquad r\approx 7\%
$$

### Spot Rate Curve (即期收益率曲线)

Computed rate for zero coupon bond.

For example, a two-year maturity bond is priced at 92.46, assumed face value of 100, the spot rate is

$$
92.46 \times (1+r)^2=100, \qquad r\approx 4\%
$$

### Forward Rate Curve (远期收益率曲线)

Computed rate for between future timestamps.

For example, again a two-year maturity bond of 100 face value is priced at 92.46 when issued.
Given that $92.46 \times (1+r)^2=100 \Rightarrow r\approx 4\%$, and trader found that the one year spot rate of this bond on the market is 3%, the forward rate is

$$
(1+0.04)^2 = (1+0.03)\times(1+r), \qquad r\approx 5\%
$$

## Curve Fitting (Interpolation)

|Method|Smoothness|Parameters|Used By|Notes|
|-|-|-|-|-|
|Hermite Interpolation|Moderate|Derivatives at knots|China bond, Ministry of Finance of the U.S.|Local slope continuity|
|Cubic B-Spline (3B)|Very High|Knot positions|FED|Global smoothness|
|Smooth Spline|Adjustable|Smoothing parameter $\lambda$|Central bank of Japan, the U.K., Canada|Data-driven, flexible|
|Nelson-Siegel (NS) and Nelson-Siegel-Svensson (NSS)|High|NS: $\beta_0,\beta_1,\beta_2,\lambda$, and NSS: NS + $\beta_3,\lambda_2$|Central bank of Italy, Germany, Finland|Economically interpretable|

Simulated results are shown as below.

<div style="display: flex; justify-content: center;">
      <img src="imgs/yield_curve_interpolation.png" width="60%" height="40%" alt="yield_curve_interpolation" />
</div>
</br>

### Interpolation In Detail

The below explanations are under the notations/assumptions:

* Let $x$ be the to-be interpolated x-axis points and $x_i$ be the existing points that map the y-axis control points $P_i=f(x_i)$.
* Let $t=\frac{x-x_i}{x_{i+1}-x_i}$ be a normalized local variable ($0<t<1$) between $(x_i, x_{i+1})$.

#### Hermite Interpolation

$H(x)$ is a polynomial that satisfies $H(x_i)=f(x_i)$ and $\frac{d}{d x_i}H(x_i)=\frac{d}{d x_i}f(x_i)$.

$$
H(x)=h_{00}(t)\cdot f(x_{i}) + h_{01}(t)\cdot f(x_{i+1}) + h_{10}(t)\cdot \frac{d}{d x_i}f(x_{i}) + h_{11}(t)\cdot \frac{d}{d x_{i+1}}f(x_{i+1})
$$

where

$$
\begin{align*}
    h_{00}(t)&=2t^3-3t^2+1 \\
    h_{01}(t)&=-2t^3+3t^2 \\
    h_{10}(t)&=t^3-2t^2+t \\
    h_{11}(t)&=t^3-t^2 \\
\end{align*}
$$

$H(x_{i+1})=f(x_i)$ mandates that the fitting curve must pass through the control points $P_i=f(x_i)$.

The gradient computation $\frac{d}{d x_i}H(x_i)=\frac{d}{d x_i}f(x_i)$ is at the discretion of user.
A common two-step differential method is

$$
\nabla f_i=\begin{cases}
    \frac{f_{i+1}-f_i}{h} &\qquad i=0 \\
    \frac{f_{i}-f_{i-1}}{h} &\qquad i=n-1 \\
    \frac{f_{i+1}-f_{i-1}}{2h} &\qquad \text{otherwise} \\
\end{cases}
$$

where $h$ is the step span.

#### Cubic B-Spline (3B)

Given four control points $P_i,P_{i+1},P_{i+2},P_{i+3}$

$$
S(t)=\sum^n_{i=0} P_i \cdot N_{i,3}(t)
$$

where

$$
N_{i,k}(t) = \frac{t - t_i}{t_{i+k} - t_i} N_{i,k-1}(t) + \frac{t_{i+k+1} - t}{t_{i+k+1} - t_{i+1}} N_{i+1,k-1}(t), \qquad k=3
$$

In detail for $k=3$,

$$
\begin{align*}
N_{i-3,3}(t)&=\frac{1}{6}(1-t)^3 \\
N_{i-2,3}(t)&=\frac{1}{6}(3t^3+6t^2+1) \\
N_{i-1,3}(t)&=\frac{1}{6}(-3t^3+3t^2+3t+1) \\
N_{i,3}(t)&=\frac{1}{6}t^3 \\
\end{align*}
$$

#### Smooth Spline

Define *natural cubic spline*:

$$
S_i(t)=a_i+b_i(t-t_i)
$$

### Nelson-Siegel (NS) and Nelson-Siegel-Svensson (NSS)

