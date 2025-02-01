import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

PERPLEXITY = 30


def pairwise_squared_distances(X):
    return squareform(pdist(X, 'sqeuclidean'))

def estimate_intrinsic_dim(X, k=2):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Ensure no division by zero: filter out points where denominator is zero
    denominator = distances[:, k-1]
    valid_mask = denominator > 0  # Mask to exclude zero denominators
    distances_valid = distances[valid_mask, :]
    
    if np.sum(valid_mask) == 0:
        raise ValueError("All denominators are zero. Check for duplicate data points.")
    
    # Compute distance ratios only for valid points
    mu = distances_valid[:, k] / distances_valid[:, k-1]
    mu = mu[~np.isnan(mu)]  # Remove NaNs (if any)
    mu = mu[~np.isinf(mu)]  # Remove Infs (if any)
    
    if len(mu) == 0:
        return 1  # Fallback to default dimension
    
    d = 1 / (np.log(mu).mean())
    return max(1, int(np.round(d)))

def pairwise_adjusted_distances(X, gamma=1.0):
    D = pairwise_squared_distances(X)
    return D ** gamma

def Hbeta(D_row, beta):
    P = np.exp(-D_row * beta)
    sum_P = np.maximum(np.sum(P), 1e-8)
    H = np.log(sum_P) + beta * np.sum(D_row * P) / sum_P
    P /= sum_P
    return H, P

def find_beta(D_row, perplexity, tol=1e-5, max_iter=50):
    desired_entropy = np.log(perplexity)
    beta_min, beta_max = -np.inf, np.inf
    beta = 1.0
    for _ in range(max_iter):
        H, P = Hbeta(D_row, beta)
        entropy_diff = H - desired_entropy
        if abs(entropy_diff) <= tol:
            break
        if entropy_diff > 0:
            beta_min = beta
            beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
        else:
            beta_max = beta
            beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
    return beta, P

def compute_p_matrix(X, perplexity=30.0, gamma=1.0):
    n = X.shape[0]
    D = pairwise_adjusted_distances(X, gamma)
    P = np.zeros((n, n))
    for i in range(n):
        D_row = D[i, :]
        mask = np.arange(n) != i
        D_row = D_row[mask]
        beta, row_P = find_beta(D_row, perplexity)
        P[i, mask] = row_P
    P = (P + P.T) / (2 * n)
    np.fill_diagonal(P, 0)
    P = np.maximum(P, 1e-12)
    return P

def compute_q_matrix(Y):
    D = pairwise_squared_distances(Y)
    Q = 1.0 / (1.0 + D)
    np.fill_diagonal(Q, 0.0)
    Q /= np.sum(Q)
    return np.maximum(Q, 1e-12)

def compute_gradient(P, Q, Y):
    n = Y.shape[0]
    grad = np.zeros_like(Y)
    for i in range(n):
        diff = Y[i] - Y
        dist = 1.0 + np.sum(diff**2, axis=1)
        inv_dist = 1.0 / dist
        pq_diff = (P[i, :] - Q[i, :]) * inv_dist
        grad[i] = 4.0 * (pq_diff @ diff)
    return grad

def tsne(X, perplexity=30.0, max_iter=1000, lr=100.0, momentum=0.8):
    d = estimate_intrinsic_dim(X)  # Estimate intrinsic dimension
    gamma = d / 2  # Power transform exponent
    P = compute_p_matrix(X, perplexity, gamma)
    n = X.shape[0]
    Y = np.random.randn(n, 2) * 1e-4
    previous_Y = Y.copy()
    gains = np.ones_like(Y)
    for it in range(max_iter):
        Q = compute_q_matrix(Y)
        kl = np.sum(np.where(P > 0, P * np.log(P / Q), 0))
        grad = compute_gradient(P, Q, Y)
        gains = (gains + 0.2) * ((grad > 0) != (previous_Y > 0)) + \
                (gains * 0.8) * ((grad > 0) == (previous_Y > 0))
        gains = np.clip(gains, 0.01, None)
        update = lr * gains * grad
        Y -= update
        Y += momentum * (Y - previous_Y)
        previous_Y = Y.copy()
        if it % 100 == 0:
            print(f"Iteration {it}, KL divergence: {kl:.4f}")
    return Y

# Example Usage
iris = load_iris()
X, y = iris.data, iris.target
Y = tsne(X, perplexity=PERPLEXITY, max_iter=1000)

# tRY with built-in TSNE
skl_tsne = TSNE(perplexity=PERPLEXITY, n_components=2)
Y_skl = skl_tsne.fit_transform(X)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap=plt.cm.Spectral)
ax.set_title('t-SNE (by custom) of Iris Dataset')

ax2 = fig.add_subplot(122)
ax2.scatter(Y_skl[:, 0], Y_skl[:, 1], c=y, cmap=plt.cm.Spectral)
ax2.set_title('t-SNE (by sklearn) of Iris Dataset')

plt.show()