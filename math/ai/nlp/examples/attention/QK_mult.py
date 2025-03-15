import numpy as np
import sys

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

Q = np.array([
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.99, 0.99, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
])

K = np.array([
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.99, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
    [0.01, 0.99, 0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01, 0.01, 0.01],
])

# Compute raw scores
S = Q @ K.T

# Compute distance penalty matrix
P = np.fromfunction(lambda i, j: 8 / (np.abs(i - j) + 1), (S.shape[0], S.shape[1]), dtype=float)

# Apply penalty by adding to S
S_prime = S * P

print(S_prime.shape)
np.savetxt(sys.stdout, S_prime, fmt='%.3f')

# Scale by sqrt(d)
d = K.shape[1]  # Dimension of key vectors
S_scaled = S / np.sqrt(d)

# Apply softmax
attention_scores = softmax(S_scaled)

# print(attention_scores)
