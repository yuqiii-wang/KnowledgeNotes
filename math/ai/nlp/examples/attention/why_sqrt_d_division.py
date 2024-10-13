import numpy as np

rand_size = 10000

mu = 0
sigma = 1
rands_1 = np.random.normal(mu, sigma, rand_size)
rands_2 = np.random.normal(mu, sigma, rand_size)
s = rands_1 * rands_2

s_mean =  np.mean(s)
print(f"mean: {s_mean}")
var_s2 = np.sum([(x-s_mean)**2 for x in s])
print(f"var: {var_s2}")
