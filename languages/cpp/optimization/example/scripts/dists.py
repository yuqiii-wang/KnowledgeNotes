import numpy as np
import time

### Compute sqrt{ (a - b)^2 }, where a and b are 3d vectors/points

a = np.ones((10000, 3))
b = np.ones((10000, 3)) + 1.0

################# FAST #################

start1 = time.time()
a2 = np.sum(a*a, axis=-1)
b2 = np.sum(b*b, axis=-1)
ab = a @ b.T # 10000 * 10000 size
# `None` is used to increase dimension, same as reshape from 1 to 2 dimensionality
dist1 = np.sqrt(a2[:, None] + b2[None, :] - 2 * ab) 
end1 = time.time()

################# SLOW #################

start2 = time.time()

a = a[:, None, :]
b = b[None, :, :]
e = a - b
dist2 = np.sqrt(np.sum(e*e, axis=-1))
end2 = time.time()

################# RESULTS #################

print("Fast solution elapsed time: " + str(end1 - start1))
print("Slow solution elapsed time: " + str(end2 - start2))