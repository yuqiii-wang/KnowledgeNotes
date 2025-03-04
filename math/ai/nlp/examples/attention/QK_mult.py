import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

Q = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.025, 0.9, 0.025, 0.025, 0.025],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
])

K = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.025, 0.9, 0.025, 0.025, 0.025],
    [0.2, 0.2, 0.2, 0.2, 0.2],
])

# Compute raw scores
S = Q @ K.T

print(S)
print(S.shape)

# Scale by sqrt(d)
d = K.shape[1]  # Dimension of key vectors
S_scaled = S / np.sqrt(d)

# Apply softmax
attention_scores = softmax(S_scaled)

# print(attention_scores)
