
import numpy as np

Q = np.array(
[[0,0.9,0.1,0,0,0],
 [0,0,1,0,0,0],
 [0,0,1,0,0,0],
 [0,0,1,0,0,0],
 [0,0,1,0,0,0]])

K = np.array(
[[0,1,0,0,0,0],
 [0,1,0,0,0,0],
 [0,0.1,0.9,0,0,0],
 [0,1,0,0,0,0],
 [0,1,0,0,0,0]])

S = Q @ K.T

print(S)