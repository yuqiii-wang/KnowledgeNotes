# Markov Chain

A Markov chain experiences transitions from one state to another according to certain probabilistic rules.

Each state's transition probability only depends on its previous state's.

![markov_chain](imgs/markov_chain.png "markov_chain")

## The Markove Property
$$
p(X_n=x_n | X_{n-1} = x_{n-1})=
p(X_n=x_n | X_{0} = x_{0}, X_{1} = x_{1}, ..., X_{n-1} = x_{n-1})
$$
where $X_i$ is a static state in Markov chain, and $x\_i$ is the $i$-th input.

## Transition Probability Maxtrix

Transition at the $i$-th time for the below graph can be expressed as

$$
p_i=
\begin{bmatrix}
      0.4 & 0.6 & 0 & 0 \\
      0.6 & 0.4 & 0 & 0 \\
      0.25 & 0.25 & 0.25 & 0.25 \\
      0 & 0 & 0 & 1
\end{bmatrix}
$$

![markov_chain](imgs/markov_chain.png "markov_chain")


## Stationary Distributions of Markov Chains

A stationary distribution of a Markov chain is a probability distribution of state $S$ that remains unchanged in the Markov chain as time progresses.

For example, Country A and country B have totaled 10 million population.
Survey found that every year, 3% of Country A nationals emigrate to country B, and 5% of Country B population emigrate to country A.
There will be a stationary distribution when Country A and Country B are stable in population size as time progresses.

$$
\begin{align*}
&
\left \{
    \begin{align*}
      x &= 0.97x + 0.05y \\
      y &= 0.03x + 0.95y \\
      10 &= x + y 
    \end{align*}
\right. \\
\Rightarrow &
\qquad x=\frac{5}{3}y \qquad x=6.25 \qquad y=3.75
\end{align*}
$$

The stationary distribution are $0.625=\frac{x}{x+y}=\frac{5/3}{5/3+1}$ and $0.375=\frac{y}{x+y}=\frac{1}{5/3+1}$ corresponding to population of 6.25 million for Country A and population of 3.75 million for Country B.

A more general solution is to treat the transitions as a transition probability matrix, then compute the eigenvalues and eigenvectors.

$$
P =
\begin{bmatrix}
      0.97 & 0.05 \\
      0.03 & 0.95
\end{bmatrix}
\qquad
\text{det}(P) \Rightarrow
\begin{vmatrix}
      0.97-\lambda & 0.05 \\
      0.03 & 0.95-\lambda
\end{vmatrix} = 0
$$

that has the solutions

$$
\lambda_1 = 1 \quad \mathbf{v}_1 = [\frac{5}{3}, 1],
\qquad
\lambda_2 = \frac{23}{25} \quad \mathbf{v}_2 = [-1, 1]
$$

where for $\lambda_1=1$ that says about probability always summed up to $1$ before and after transition, its corresponding eigenvectors are $0.625=\frac{5/3}{5/3+1}$ and $0.375=\frac{1}{5/3+1}$.

In fact, $\lim_{n \rightarrow \infty} P^n$ represents that after some steps as $n$ increases, the stationary probability distribution result is converged.
For example, $P^{10} = \begin{bmatrix} 0.78789567 & 0.35350722 \\ 0.21210433 & 0.64649278 \end{bmatrix}$ and $P^{100} = \begin{bmatrix} 0.6250897 & 0.62485049 \\ 0.3749103 & 0.37514951 \end{bmatrix}$ shows that as time progresses, population of the two countries on the $10$-th and $100$-th year results are converged to $[0.625, 0.375]$.

## Markov Decision Process

*Markov decision process* added probability transition *action* $a_k$ between states $s_i$ and $s_j$.
The transition probability can be expressed as $P_a(s_i, s_j)=P(s_{t+1}=s | s_t=s, a_t=a)$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/markov_decision_proc.png" width="30%" height="30%" alt="markov_decision_proc" />
</div>
</br>

*Markov decision process* added probability 