# Normal Distribution

Normal distribution $X \sim N(\mu, \sigma^2)$ has probability density function

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp \Big( -\frac{1}{2} \big( \frac{x-\mu}{\sigma} \big)^2 \Big)
$$

Consider a standard normal distribution ($\mu=0$ and $\sigma=1$):
$$
f(x) = \frac{1}{\sqrt{2\pi}} \exp \Big( -\frac{1}{2} x^2 \Big)
$$

Set $Z = \frac{X-\mu}{\sigma}$, so that a normal distribution can be transformed to a standard one.

After converting to a standard one, can use a table to find the corresponding cumulative area.
Notice that the whole area of a standard normal distribution is $1$, so that the cumulative area represents the probability.

For example, to find $Z=1.09$, can search in the table below by $1.09 = 1.0 + 0.09$

<div style="display: flex; justify-content: center;">
      <img src="imgs/normal_dist_table.png" width="50%" height="30%" alt="normal_dist_table" />
</div>
</br>

The corresponding area is 0.8621, 

<div style="display: flex; justify-content: center;">
      <img src="imgs/normal_dist_1.09_z.png" width="30%" height="20%" alt="normal_dist_1.09_z" />
</div>
</br>

## Normal Approximation to the Binomial Distribution

For example, given $X \sim \text{Binomial}(4000, 0.8)$ (4000 trials where there are $p=0.8$ probability of hitting true against 0.2 of hitting false), to compute the probability $P(X > 3500)$, 
a naive solution is to sum all $P(X=3501) + P(X=3502) + ... + P(X=4000)$.

However, it can be approximated by converting to a standard normal distribution: $\mu=np=3200$ and $\sigma^2=np(1-p)=640$.
Let $Y$ be this approximation: $Y \sim N(3200, 640)$.
Then 


## Studentized Range Distribution (q-Distribution)

Studentized range distribution (q-distribution) selectively seeks extreme differences in sample data, rather than only sampling randomly.

Let $X_{ij}$ represent the $j$-th observation in the $i$-th population (there are a total of $k$ population groups) having a normal distribution with mean $\mu_i$ and variance $\sigma_i^2$ (by equal variance assumption, there is $\sigma^2=\sigma^2_1=\sigma^2_2=...=\sigma^2_i...=\sigma^2_k$). 
There is $X_{ij} \sim N (\mu_i, \sigma^2)$.

Rather than randomly selecting samples from populations, q-distribution find the largest sample mean $\overline{X}_{max}$ and smallest sample mean $\overline{X}_{min}$.
Set $s^2$ is the pooled sample variance from these samples.

$$
q = \frac{\overline{X}_{max}-\overline{X}_{min}}{\frac{s}{\sqrt{n}}}
$$