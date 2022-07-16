# Leibniz integral rule

Leibniz integral rule describes the derivative rule of an integral with respect to two diff variables.

Given an integral form
$$
\int^{b(x)}_{a(x)} f(x,t) dt
$$

for $-\infty < a(x) < b(x) < +\infty$, its derivative is
$$
\frac{d}{dx} 
\bigg(
    \int^{b(x)}_{a(x)} f(x,t) dt
\bigg)
=
\\
f\big(x,b(x)\big)\cdot \frac{d \space b(x)}{dx}
-
f\big(x,a(x)\big)\cdot \frac{d \space a(x)}{dx}
+
\int^{b(x)}_{a(x)} \frac{\partial}{\partial x} f(x,t) dt
$$

If $a(x)=c_a$ and $b(x)=c_b$, where $c_a, c_b$ are constant, $f\big(x,b(x)\big)\cdot \frac{d \space b(x)}{dx}$ and $f\big(x,a(x)\big)\cdot \frac{d \space a(x)}{dx}$ are zeros, there is
$$
\frac{d}{dx} 
\bigg(
    \int^{c_b}_{c_a} f(x,t) dt
\bigg)
=
\int^{c_b}_{c_a} \frac{\partial}{\partial x} f(x,t) dt
$$