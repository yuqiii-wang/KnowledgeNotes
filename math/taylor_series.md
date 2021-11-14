# Taylor Series

A function is an infinite sum of terms that are expressed in terms of the function's derivatives at a single point.

If $f(x)$ is given by a convergent power series in an open disk (or interval in the real line) centred at $b$ in the complex plane, it is said to be analytic in this disk. Thus for $x$ in this disk, $f(x)$ is given by $a$ convergent power series 

$$f(x) = \sum^{\inf}_{n=0}a_n(x-b)^n$$

Differentiating by $x$ the above formula $n$ times, then setting $x = b$ gives: 

$$ \frac{f^{(n)}(b)}{n!} = a_n$$

A function is analytic in an open interval/disk (interval for real and disk for a complex plane)centred at b if and only if its Taylor series converges to the value of the function at each point of the disk. 

Taylor series diverges at $x$ if the distance between $x$ and $b$ is larger than the radius of convergence (defined by $|\frac{a_{n+1}}{a_n}|<1$). 