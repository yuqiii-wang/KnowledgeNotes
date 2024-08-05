# Random/Stochastic Process

A stochastic process is defined as a collection of random variables used as mathematical models to simulate a system or a phenomenon.


## Law of large numbers

The average of the results obtained from a large number of trials should be close to the expected value and tends to become closer to the expected value as more trials are performed.

In other words, more trials, more accurate the mean value.

## Bessel's Correction

Bessel's correction is the use of $n − 1$ as the sample number instead of $n$ in the formula for the sample variance and sample standard deviation.

A naive approach to variance is
$$
\sigma^2 = \overline{(x^2)} - (\overline{x})^2 =
\frac{
    \sum_{i=1}^n x_i^2 - \frac{1}{n}\Big(\sum_{i=1}^n x_i \Big)^2}{ n }
$$

By Bessel's correction, there is
$$
s^2 = \frac{n}{n-1} \Bigg( \frac{1}{n}\sum_{i=1}^n x_i^2 - \Big( \frac{1}{n}\sum_{i=1}^n x_i \Big)^2 \Bigg)
$$

One can understand Bessel's correction as the degrees of freedom in the residuals vector (residuals, not errors, because the population mean is unknown):
$$
\{ x_1-\overline{x}, x_2-\overline{x}, ..., x_n-\overline{x} \}
$$
where $\overline{x}$ is the sample mean.
For $n$ independent observations in the sample, there are only $n − 1$ independent residuals, as they sum to $0$. 

### Source of Inspiration and Bias

The reason why there are only $n − 1$ independent residuals is that, for example, there is only one sample point $\bold{x} = \{ x_1 \}$.
The mean is $\overline{x} = x_1$ and residual is always $x_1 - \overline{x} = 0$, that the residual is not independent when $n=1$.

As a result, residual is always has degree of freedom of $n-1$.

## Central Limit Theorem

The *central limit theorem* (CLT) establishes that, in many situations, for identically distributed independent samples, the standardized sample mean tends towards the standard normal distribution even if the original variables themselves are not normally distributed.

$\bold{x} = \{ \bold{x}_1, \bold{x}_2, ..., \bold{x}_n, ... \}$ are random samples having a mean $\mu$ and finite variance $\sigma^2$.
If $\overline{\bold{x}}_n$ is the sample mean of the first $n$ samples, then $Z=\lim_{n \rightarrow \infty} \Big( \frac{\overline{\bold{x}}_n - \mu}{\frac{1}{\sqrt{n}} \sigma} \Big)$ is a standard normal distribution (having $\mu_{Z}=0$).
