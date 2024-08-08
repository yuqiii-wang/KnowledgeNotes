import numpy as np

#==================
# for matrix multiplication

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

element_wise_product = A * B
print(element_wise_product)

matrix_product = np.dot(A, B)
print(matrix_product)

#==================
# for vector multiplication

v1 = np.array([1, 2])
v2 = np.array([3, 4])

dot_product = np.dot(v1, v2)
print(dot_product)

cross_product = np.cross(v1, v2)
print(cross_product)

outer_product = np.outer(v1, v2)
print(outer_product)