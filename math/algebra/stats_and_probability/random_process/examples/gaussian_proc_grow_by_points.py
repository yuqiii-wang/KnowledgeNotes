from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)

def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()

train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
train_y = y(train_X, noise_sigma=1e-4)
test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

gpr = GPR()
fig = plt.figure(figsize=(12, 4))
plot_count = 0
for idx in range(len(train_X)+1):
    if idx == 2 or idx == 4:
        continue
    plot_count += 1
    gpr.fit(train_X[:idx], train_y[:idx])
    mu, cov = gpr.predict(test_X)
    test_y = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    ax_idx = fig.add_subplot(1, 4, plot_count)
    ax_idx.set_title(f"{idx} sample points\n$l={gpr.params["l"]}$ $\sigma_f={gpr.params["sigma_f"]}$")
    ax_idx.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
    ax_idx.plot(test_X, test_y, label="mean")
    ax_idx.scatter(train_X[:idx], train_y[:idx], label="observation", c="red", marker="x")
    ax_idx.legend()
plt.show()
