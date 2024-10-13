import numpy as np

# Dimensions
dim = 6         # Dimension of each token
learning_rate = 0.0005
fc_output_dim = 6  # Output dimension of the fully connected layer
# Training loop
num_epochs = 1000

# Sample input data with polarized values
source_X = [[0.5,0.5,0,0,0,0] if idx % 10 == 0 else 
            [0,0,0.25,0.25,0.25,0.25] if idx % 10 == 1 else 
            [0,0,0,0,0,0] if idx % 10 == 2 else 
            [0,0,0,0,0,0] if idx % 10 == 3 else 
            [0,0,0,0,0,0] if idx % 10 == 4 else 
            [0,0,0,0,0,0] if idx % 10 == 5 else 
            [0,0,0,0,0,0] if idx % 10 == 6 else 
            [0,0,0,0,0,0] if idx % 10 == 7 else 
            [0,0,0,0,0,0] if idx % 10 == 8 else 
            [0,0,0,0,0,1] for idx in range(50)]
num_tokens = len(source_X)

X = np.array(source_X)
print(X[:10])

Y = np.roll(X, 0, axis=0)
print(Y[:10])

target_output = np.roll(Y, 0, axis=0)
print(target_output[:10])

# Initialize weights
W_Q = np.random.rand(dim, dim)
W_K = np.random.rand(dim, dim)
W_V = np.random.rand(dim, dim)

# Initialize weights for the fully connected layer
W_fc = np.random.rand(dim, fc_output_dim)  # Fully connected layer weight
b_fc = np.random.rand(fc_output_dim)       # Bias for the fully connected layer

# Initialize parameters for layer normalization
gamma_attention = np.ones(dim)
beta_attention = np.zeros(dim)
gamma_fc = np.ones(fc_output_dim)
beta_fc = np.zeros(fc_output_dim)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return softmax_result

def cross_attention(X, Y):
    Q = X @ W_Q  # Query
    K = Y @ W_K  # Key
    V = Y @ W_V  # Value

    # Calculate attention scores
    scores = Q @ K.T / np.sqrt(dim)  # Scale the dot product
    attention_weights = softmax(scores)  # Apply softmax to get attention weights
    output = attention_weights @ V  # Weighted sum of values

    return output, attention_weights, Q, K, V

def fully_connected_layer(output):
    # Apply fully connected layer
    return output @ W_fc + b_fc

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / (std + eps) + beta

# Define a simple loss function (mean squared error)
def loss_function(predicted, target):
    return np.mean((predicted - target) ** 2)

for epoch in range(num_epochs):

    attention_output, attention_weights, Q, K, V = cross_attention(X, Y)

    # Forward pass through fully connected layer
    fc_output = fully_connected_layer(attention_output)

    # Layer normalization for fully connected output
    fc_output = layer_norm(fc_output, gamma_fc, beta_fc)

    # Compute loss
    loss = loss_function(fc_output, target_output)

    d_fc_output = 2 * (fc_output - target_output) / num_tokens  # Derivative of loss w.r.t output

    # Backpropagation
    d_W_fc = attention_output.T @ d_fc_output  # Gradient of W_fc
    d_b_fc = np.sum(d_fc_output, axis=0)  # Gradient of b_fc

    # Compute gradients for attention weights
    d_attention_weights = d_fc_output @ V.T  # Correct shape for backpropagation
    d_attention_weights = d_attention_weights * attention_weights * (1 - attention_weights)  # Backprop through softmax

    # Update weights
    W_fc -= learning_rate * d_W_fc
    b_fc -= learning_rate * d_b_fc

    # Update weights (simplified)
    W_Q_grad = (X.T @ (d_attention_weights @ (K / np.sqrt(dim)))).mean(axis=0)
    W_K_grad = (Y.T @ (d_attention_weights.T @ (Q / np.sqrt(dim)))).mean(axis=0)
    W_V_grad = (Y.T @ d_attention_weights).mean(axis=1)

    W_Q -= learning_rate * W_Q_grad + np.random.rand(dim, dim)*0.0005-0.00025
    W_K -= learning_rate * W_K_grad + np.random.rand(dim, dim)*0.0005-0.00025
    W_V -= learning_rate * W_V_grad + np.random.rand(dim, dim)*0.0005-0.00025

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Sample input data with polarized values
test_X = [[0.49,0.49,0.02,0,0,0] if idx % 10 == 0 else 
            [0,0,0.25,0.25,0.25,0.25] if idx % 10 == 1 else 
            [0,0,0,0,0,0] if idx % 10 == 2 else 
            [0,0,0,0,0,0] if idx % 10 == 3 else 
            [0,0,0,0,0,0] if idx % 10 == 4 else 
            [0,0,0,0,0,0] if idx % 10 == 5 else 
            [0,0,0,0,0,0] if idx % 10 == 6 else 
            [0,0,0,0,0,0] if idx % 10 == 7 else 
            [0,0,0,0,0,0] if idx % 10 == 8 else 
            [0,0,0,0,0.01,0.99] for idx in range(20)]
X = np.array(test_X)
Y = X
output, attention_weights, Q, K, V = cross_attention(X, Y)
print(output)

print("""
The above attention attempts to learn a sequence of tokens
that repeat every ten occurrences.
The result shows that the 1st and 2nd rows are distinctively diff
from all-zeros rows; the 10th row is also obviously diff from other rows.
"""
)