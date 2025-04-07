import numpy as np

# the sum of a num of random vars is the sqrt of its variance
# For example, if X \sim N(0,1), the E(X)=\sqrt(n * sigma^2)

np.sum([np.random.normal(0,1) for _ in range(10000)])
