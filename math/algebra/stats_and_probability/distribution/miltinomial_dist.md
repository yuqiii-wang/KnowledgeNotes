# Multinomial  Distribution

## Binomial Distribution

Binomial distribution with parameters $n$ and $p$ is the discrete probability distribution of the number of successes in a sequence of $n$ independent experiments, each asking a yes–no question, and each with its own Boolean-valued outcome: success (with probability $p$) or failure (with probability $q = 1 − p$).

A single success/failure experiment is also called a Bernoulli trial or Bernoulli experiment, and a sequence of outcomes is called a Bernoulli process; for a single trial, i.e., $n = 1$, the binomial distribution is a Bernoulli distribution.

### Permutation and Combination

* Permutation

Such as $n=16,k=3$, the $16$ options are not retrievable (once adopted one, there are $15$ left). Try for $3$ times.

$$
\begin{align*}
\frac{n!}{(n-k)!}
&=
\frac{16!}{(16-3)!}
\\ &=
\frac{16 \times 15 \times 14 \times 13 \times ...}
{13 \times 12 \times 11 \times 10 \times ...}
\\ &=
16 \times 15 \times 14
\end{align*}
$$

* Combination

Such as $n=16,k=3$, the $16$ options are not retrievable (once adopted one, there are $15$ left). Besides, the option choices are orderless, that first picking option $o_1$ then option $o_2$, is identical to first picking $o_2$ then $o_1$. Try for $3$ times.

$$
\begin{align*}
\frac{n!}{k!(n-k)!}
&=
\frac{16!}{3!(16-3)!}
\\ &=
\frac{16 \times 15 \times 14 \times 13 \times ...}
{(3\times 2\times 1) \times(13 \times 12 \times 11 \times 10 \times ...)}
\\ &=
\frac{16 \times 15 \times 14}{3\times 2\times 1}
\end{align*}
$$

* Permutation and Combination

Together, they are denoted as 

$$
\left(
\begin{array}{c}
n
\\
k
\end{array}
\right)
=
\frac{n!}{k!(n-k)!} 
$$

### Example

Random variable $X$ follows the binomial distribution with parameters $n \in \mathbb{N}$  and $p \in [0,1]$, we write $X \sim B(n, p)$. Now having $k$ success trails, there is

$$
\left(
\begin{array}{c}
n
\\
k
\end{array}
\right)
p^k(1-p)^{n-k}
$$

## Multinomial distribution

Multinomial distribution is a generalization of the binomial distribution. There are $k$ outcomes rather than boolean results as in Binomial Distribution.

There are $k$ outcomes: $x_1, x_2, ..., x_k$, correspodning to $k$ probabilities $p_1, p_2, ..., p_k$. The number of trials is $n$.
$$
f(x_1, x_2, ..., x_k; n; p_1, p_2, ..., p_k)
\\ 
\quad\quad\quad\quad\quad=
\begin{cases}
\frac{n!}{x_1!...x_k!}p_1^{x_1}p_2^{x_2}...p_k^{x_k} \quad \sum^k_{i=1}x_i=n
\\
0 \quad\quad\quad\quad\quad\quad\quad\quad \text{otherwise}
\end{cases}
$$