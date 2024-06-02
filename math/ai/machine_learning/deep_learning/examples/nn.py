
import numpy as np

# Sigmoid function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation(x, w, b):
    z = np.dot(x, w) + b
    y_hat = sigmoid(z)
    return y_hat, z

# Backward propagation
def backward_propagation(x, y, y_hat, z):
    m = x.shape[0] # num opf samples
    dz = (y_hat - y) * y_hat * (1-y_hat)
    dw = np.dot(x.T, dz)
    db = np.sum(dz, axis=0) / m
    return dw, db

# Example usage:
np.random.seed(42)

x = np.random.randn(5, 3)  # 5 samples, 3 features
y = np.random.randn(5, 2)  # 5 samples, 2 outputs

w = np.random.randn(3, 2)  # 3 features, 2 outputs
b = np.random.randn(2)     # 2 bias terms

learning_rate = 0.05

y_hat, z = forward_propagation(x, w, b)
loss = np.mean((y_hat - y) ** 2)

epoch = 0
while loss > 1e-5 and epoch < 1000:
    # Forward propagation
    y_hat, z = forward_propagation(x, w, b)

    # Compute loss (MSE)
    loss = np.mean((y_hat - y) ** 2)

    # Backward propagation
    dw, db = backward_propagation(x, y, y_hat, z)

    w -= learning_rate * dw
    b -= learning_rate * db

    epoch += 1
    if epoch % 100 == 0:
        print(f"epoch: {epoch}")
        print(f"w: {w}")
        print(f"b: {b}")
        print(f"Loss: {loss}")