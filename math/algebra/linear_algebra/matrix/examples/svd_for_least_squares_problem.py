import numpy as np

A = np.array([[1,0], [0,1], [1,1], [0,0]])
b = np.array([0.0, 0.0, np.sqrt(2)/2, np.sqrt(2)/2])

### SVD Formula
U, S, Vh = np.linalg.svd(A, full_matrices=False)
y = np.matmul(U.transpose(), b) / S
x = np.matmul(Vh.transpose(), y)

print(x)

### Verify
verify = np.linalg.lstsq(A, b)
verify_x = verify[0]
verify_residual = verify[1]
verify_rankA = verify[2]
verify_sigma = verify[3]

print(verify_x)