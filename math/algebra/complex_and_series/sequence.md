# Sequence

## Sum of a Finite Geometric Sequence

For a geometric sequence $a_n=k r^{n-1}$ conditional on $r \ne 1$, its sum is
$$
\begin{align*}
S_n &= 
k + kr + kr^2 + ... + kr^{n-1}
\\ &=
k \space \frac{r-1}{r-1} (1+r+r^2+...+r^{n-1})
\\ &=
k \space \frac{1}{r-1}
\big((r+r^2+...+r^{n})-(1+r+r^2+...+r^{n-1})\big)
\\ &=
k \frac{1}{r-1}(r^n-1)
\\ &=
k\frac{1-r^n}{1-r}
\end{align*}
$$