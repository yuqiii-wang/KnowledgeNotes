# Differential Equation

## Linear Differential Equation

A linear differential equation is a differential equation that is defined by a linear polynomial in the unknown function and its derivatives, that is an equation of the form
$$
b(x) = a_0(x)y+a_1(x)y'+a_2(x)y''+...+a_n(x)y^{(n)}
$$
where 

$a_i(x)$ and $b(x)$ are differentiable functions that do not need to be linear,

$y^{(i)}$ is successive derivative of an unknown function $y$ of the variable $x$.

Linearity here indicates sum of $a_i(x)y^{(i)}$.

### Homogeneous lLnear Differential Equation

A homogeneous linear differential equation has constant coefficients if it has the form

$$
0 = a_0y+a_1y'+a_2y''+...+a_ny^{(n)}
$$
where $a_i$ is a (real or complex) number.

By $exp$, the $n$-derivative of $e^{cx}$ is $c^n e^{cx}$, hence
$$
0 = a_0e^{cx}+a_1ce^{ax}+a_2c^2e^{cx}+...+a_nc^ne^{cx}
$$

Factoring out $e^{cx}$ since it will never be zero, and here derives characteristic equation
$$
0 = a_0+a_1c+a_2c^2+...+a_nc^n
$$

Let $z$ be a complex number, there is 
$$
0 = a_0+a_1z+a_2z^2+...+a_nz^n
$$
