# Bayes' Theorem

## Concepts

*Prior* expresses one's beliefs about this quantity before some evidence is taken into account, denoted $p(\theta)$

*Posterior* expresses condition's/parameter $\theta$ probability given observations/evidence $X$, denoted $p(\theta|X)$

*Likelihood* is the probability of an event $x$ given condition's/parameter $\theta$, denoted as $p(x|\theta)$

$$
p(\theta|x)=\frac{p(x|\theta)p(\theta)}{\int{p(x|\theta)p(\theta)d\theta}}
$$

## Questions

### Q1
When a lady is diagonised having a disease, the doctor told her that in general the probability of having the disease is 0.001, and ordered a professional test with a device (this device has accuracy of 0.99). What is the probability of having the disease given a positive test result.

0.001 * 0.99 + 0.999 * 0.01

### Q2
An entomologist spots what might be a rare subspecies of beetle, due to the pattern on its back. In the rare subspecies, 98% have the pattern, or P(Pattern | Rare) = 98%. In the common subspecies, 5% have the pattern. The rare subspecies accounts for only 0.1% of the population. How likely is the beetle having the pattern to be rare, or what is P(Rare | Pattern)?

P(Rare | Pattern) 
                = P(Pattern | Rare) * P(Rare) / P(Pattern)
                = P(Pattern | Rare) * P(Rare) / (P(Pattern | Rare) * P(Rare) + P(Pattern | Common) * P(Common))
                = (0.98 * 0.001) / (0.98 * 0.001 + 0.05 * 0.999)