# Neural Tangent Kernel (NTK)

Neural Tangent Kernel (NTK) is a study of the **dynamics** of neural networks during training by **gradient descent** on the assumption that the network is **infinitely wide**.
It reveals what the core invariant behavior of the network is as parameters change/update in training process.

References:

* https://www.eigentales.com/NTK/
* https://zhuanlan.zhihu.com/p/110853293
* https://zhuanlan.zhihu.com/p/123047060
* https://zhuanlan.zhihu.com/p/139678570

Given a deep (with $l$ layers) and very wide (dimension is $d\rightarrow\infty$) neural network where each layer is parameterized by $\mathbf{\theta}_l\in\mathbb{R}^{d}$, and denote activation functions as $\sigma_l$.

$$
f_{\mathbf{\theta}}(\mathbf{x})=
\underbrace{\sigma_{l}\big(\qquad\qquad...\qquad\qquad
\underbrace{\sigma_{2}(W_{2}
\underbrace{\sigma_{1}(W_{1}\mathbf{x}+\mathbf{b}_{1})}_{\text{layer }1}
+\mathbf{b}_{2})}_{\text{layer }2}
\big)}_{\text{layer }l}
$$

The NTK kernel is defined as

$$
\kappa(\mathbf{x}, \mathbf{x}') = \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top\big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x}')\big)
$$

where $\mathbf{x}, \mathbf{x}'\in\mathbb{R}^d$ are the input vectors, and $\mathbf{\theta}\in\mathbb{R}^d$ is the parameter vector for the neural network $f_{\mathbf{\theta}}(.)$.

NTK major conclusions are

* For an infinite-width network, if parameter $\mathbf{\theta}$ is initialized from a certain distribution (e.g., Gaussian distribution), then the kernel $\kappa_{\mathbf{\theta}}(\mathbf{x}, \mathbf{x}')$ is deterministic (invariant to individual parameter changes)
* NTK kernel $\kappa_{\mathbf{\theta}}(\mathbf{x}, \mathbf{x}')=\kappa_{\mathbf{\theta}_0}(\mathbf{x}, \mathbf{x}')$ is invariant within an arbitrary training iteration step $\Delta t$ equal to before the iteration kernel $\kappa_{\mathbf{\theta}_0}(\mathbf{x}, \mathbf{x}')$.
* An infinite-width network is linear.
* Eigenvalues of NTK matrix determines effectiveness of learning rate $\eta$ setup in training a neural network.

## NTK Intuition

For a neural network $f_{\mathbf{\theta}}(\mathbf{x})$ parameterized by $\mathbf{\theta}\in\mathbb{R}^d$ given input $\{\mathbf{x}_i\}_{i=1}^n$,
at any iteration, $f_{\mathbf{\theta}_t}(\mathbf{x})$ can be modeled as

$$
f_{\mathbf{\theta}_t}(\mathbf{x}) \approx f_{\mathbf{\theta}_0}(\mathbf{x}) + \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})\big)^{\top} (\mathbf{\theta}_t - \mathbf{\theta}_0)
$$

where $f_{\mathbf{\theta}_0}(\mathbf{x})$ is the previous iteration updated network taken as to-start-update network for the current iteration step, NOT the start of training process.

This first-order Taylor expansion indicates that

* The gradient at initialization $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$ is invariant at the current iteration.

### NTK Intuition by Terminology

The terminology "Neural Tangent Kernel" can be explained as

* Neural: neural network parameterized by $\mathbf{\theta}\in\mathbb{R}^d$
* Tangent: approximate the gradient by linearity by $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$
* Kernel: the function that reveals the core invariant behavior of the network

### NTK Intuition About the "Dynamics" of Gradient Descent

The NTK kernel $\kappa_{\mathbf{\theta}}(\mathbf{x}, \mathbf{x}')$ is derived from $\frac{d f_{\mathbf{\theta}}(\mathbf{x})}{dt}$ which describes how $f_{\mathbf{\theta}}$ changes as $\mathbf{\theta}$ updates with an iteration step $\Delta t$.

Denote the before update network as $f_{\mathbf{\theta}_0}$, and after update as $f_{\mathbf{\theta}_t}$.

The dynamics describes how $f_{\mathbf{\theta}_0}\rightarrow f_{\mathbf{\theta}_t}$ in ONE training step.

This means $f_{\mathbf{\theta}_0}$ takes from previous update output as input, and the kernel $\kappa_{\mathbf{\theta}}(\mathbf{x}, \mathbf{x}')$ revealed covariance distribution does not only concern how parameters are initialized, but also the WHOLE training process in which each iteration input/output needs to get aligned with the same distribution covariance.

This intuition is similar to a generic dynamic system that uses first-order gradient to approximate how system progresses given multiple iterative updates.
Each iteration $f_{\mathbf{\theta}_0}\rightarrow f_{\mathbf{\theta}_t}$ is considered that $f_{\mathbf{\theta}}$ is changed from last time system update, and in the current iteration $f_{\mathbf{\theta}_0}$ is used to represent this iteration system state.
When the update step is small enough $\Delta t\rightarrow 0$, the dynamic system is considered continuous.

### NTK Intuition by Effects of Different Initialization Distributions

NTK kernel is essentially a recursive operation multiplying the covariance matrix of the previous layer,
so that covariance matrix of each layer parameters matters.

Typically, the initialization distribution of the parameters follows Gaussian distribution $\mathbf{\theta}_0\sim\mathcal{N}(0, \sigma^2)$,

If by uniform distribution $\mathbf{\theta}_0\sim\mathcal{U}(-a, a)$, the variance is $\frac{a^2}{3}$, and the convergence is different from Gaussian distribution induced initialization.
As a result, the found optimal $\mathbf{\theta}^*$ is different from that of Gaussian distribution induced initialization.

However, if set Gaussian distribution initialization by $\mathbf{\theta}_0\sim\mathcal{N}(0, \frac{a^2}{3})$ where the variance is the same as the uniform distribution, the convergence behavior (determined by NTK kernel) is the same as the uniform distribution initialization.

### NTK Intuition For Effectiveness of Learning Rate Setup

Abstract: the eigenvalues of the NTK matrix determine the effectiveness of the learning rate setup.
By studying the max eigenvalue $\lambda_{\max}$ of the NTK matrix, the convergence behavior can be determined.

Consider a simple network: $f(\mathbf{x})=\frac{1}{\sqrt{d}} \mathbf{w}_i^{\top} \mathbf{w}_j \mathbf{x}$ (a very wide network is essentially a linear model, hence this simple network is representative of a typical neural network).

For the parameter initialization follow Gaussian distribution $\mathbf{\theta}_0\sim\mathcal{N}(0, \frac{1}{\sqrt{d}})$, to simplify the network, to a $1$-d linear model $f(x)=\frac{1}{\sqrt{d}} w_i^{\top} w_j x$, there should be $w_i, w_j \sim\mathcal{N}(0, 1)$.
Here sets $w_i=w_j=w$ for simplicity.

Let $\eta$ be the learning rate, and let input $x=1$ and output $y=0$, loss be MSE, compute the NTK kernel:

$$
\begin{align*}
    \kappa_t(1,1) &= \big((\frac{\partial f}{\partial w_i})^{\top}(\frac{\partial f}{\partial w_i})+(\frac{\partial f}{\partial w_j})^{\top}(\frac{\partial f}{\partial w_j})\big) \\\\
    &= \big((\frac{1}{\sqrt{d}}w_j)^{\top}(\frac{1}{\sqrt{d}}w_j)+(\frac{1}{\sqrt{d}}w_i)^{\top}(\frac{1}{\sqrt{d}}w_i)\big) \\\\
    &= \frac{1}{d}\big(||w_i||^2_2+||w_j||^2_2\big) \\\\
    &=\frac{1}{d}\big(||w||^2_2+||w||^2_2\big) && \text{NTK has the same elements}\\\\
    &=\lambda_t && \text{One-element matrix' eigenvalue is the element itself.}
\end{align*}
$$

Compute the updates of the parameters:

$$
\Delta w_i=-\eta w_j\frac{f}{\sqrt{d}}, \quad \Delta w_j=-\eta w_i\frac{f}{\sqrt{d}}
$$

Compute the updates of the function and eigenvalues:

$$
\begin{align*}
    f_{t+1}&=\frac{1}{\sqrt{d}} (w_i+\Delta w_i)^{\top} (w_j+\Delta w_j) \\\\
    &=\frac{1}{\sqrt{d}} (w_i-\eta w_j\frac{f_t}{\sqrt{d}})^{\top} (w_j-\eta w_i\frac{f_t}{\sqrt{d}}) \\\\
    &= \frac{1}{\sqrt{d}} (||w||^2_2-2\eta ||w||^2_2 \frac{f_t}{\sqrt{d}}+(\eta\frac{f_t}{\sqrt{d}})^2||w||^2_2) & \text{substitute with } f_t=\frac{1}{\sqrt{d}}||w||^2_2\\\\
    &= f_t-\frac{2}{d}\eta||w||^2_2f+\eta^2\frac{f_t^2}{d}f_t \\\\
    &= (1-\frac{2}{d}\eta||w||^2_2+\eta^2\frac{f_t^2}{d})f_t \\\\
    &= (1-\eta\lambda+\eta^2\frac{f_t^2}{d})f_t \\\\
\end{align*}
$$

Recall that training by gradient descent if NTK is stable is

$$
\begin{align*}
    f_{t+1} &= f_{t} - \eta\nabla_{\mathbf{w}}f_{t}\cdot f_t \\\\
    &= f_{t} - \eta\kappa(1, 1) f_t \\\\
    &= (1 - \eta\lambda) f_t \\\\
\end{align*}
$$

Compare $(1-\eta\lambda+\eta^2\frac{f_t^2}{d})f_t$ vs $(1 - \eta\lambda) f_t$, the missing $\eta^2\frac{f_t^2}{d}\approx 0$ is attributed to the NTK stability for $d\rightarrow\infty$ (not the case in this $d=1$ example).

Compute the eigenvalue update of the NTK kernel:

$$
\begin{align*}
    \kappa_{t+1}(1,1) &= \big((\frac{\partial f_{t+1}}{\partial w_i})^{\top}(\frac{\partial f_{t+1}}{\partial w_i})+(\frac{\partial f_{t+1}}{\partial w_j})^{\top}(\frac{\partial f_{t+1}}{\partial w_j})\big) \\\\
    &=\lambda_t+\eta^2\frac{f_t^2}{d}(\eta\lambda_t-4) \\\\
    &=\lambda_{t+1}
\end{align*}
$$

To study the convergence behavior, here discusses the progress of $\kappa_{t}(1,1)$ to $\kappa_{t+1}(1,1)$ and $f_t$ to $f_{t+1}$ as example.
For in this example the dimensionality is $d=1$, set $\lambda_{\max}=\lambda_t$.

* Lazy Phase: $\eta<\frac{2}{\lambda_{\max}}$

Let $f_t$ be the loss function, and it should see monotonic decrease, i.e., $f_{t+1}<f_t$.
Assume NTK kernel is stable, there should be $\frac{f_{t+1}}{f_t}<1 \Rightarrow 1-\eta\lambda_{\max}<1$.

The linearized dynamics is $f_{t}=f_0(1-\eta\lambda_{\max})^t$.
Hence, for convergence the coefficient $|1-\eta\lambda_{\max}|<1 \Rightarrow \eta<\frac{2}{\lambda_{\max}}$.

* Catapult Phase: $\frac{2}{\lambda_{\max}}<\eta<\frac{4}{\lambda_{\max}}$

For $|1-\eta\lambda_{\max}|>1$, the dynamics $f_{t}=f_0(1-\eta\lambda_{\max})^t$ experiences oscillatory convergence (output magnitude grows exponentially).

As an example, let $\eta=\frac{3}{\lambda_t}$, the NTK kernel is monotonically decreasing $\lambda_{t+1}=\lambda_t+\frac{9f_t^2}{d\lambda_t^2}(3-4)=\lambda_t-\frac{9f_t^2}{d\lambda_t^2}$.
Although $f_{t}=f_0(1-\eta\lambda_{\max})^t$ grows exponentially, the growth rate (determined by NTK kernel $\lambda_{t}\rightarrow\lambda_{t+1}$) dampens over time.

When NTK kernel is small enough to $\lambda_t\approx\frac{2}{\eta}$, the loss growth is reverted $f_{t}=f_0(1-\eta\lambda_{\max})^t$ that it becomes monotonically decreasing.

Large learning rate in catapult phase leads to greater generalization than a small one from lazy phase.

* Divergent Phase: $\eta>\frac{4}{\lambda_{\max}}$

For the NTK kernel be stable, there should be $\lambda_{t+1}\approx\lambda_t$, the residual $\eta^2\frac{f_t^2}{d}(\eta\lambda_t-4)\approx 0 \Rightarrow \eta\lambda_t\approx 4$.
Large learning rate $\eta>\frac{4}{\lambda_{\max}}$ leads to kernel divergent.

## NTK Proof Steps

To prove the NTK theorem, the steps are

1. Show input/output to next layer is of a Gaussian process (just by studying one layer could be enough be applied (by chain rule) to all layers)
2. In a training step, the Jacobian is stable, i.e., $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_t}(\mathbf{x})\approx\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$.
3. Show the kernel is linear

### NTK Definition and Deduction by Gradient Flow

Here shows how NTK is derived.

Consider a neural network $f_{\mathbf{\theta}}(\mathbf{x})$ parameterized by $\mathbf{\theta}\in\mathbb{R}^d$ trained on a dataset $\mathcal{D}=\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with mean squared error (MSE) loss:

$$
\mathcal{L}(\mathbf{\theta}) = \frac{1}{n} \sum_{i=1}^n \frac{1}{2}\big(y_i - f_{\mathbf{\theta}}(\mathbf{x}_i)\big)^2
$$

At an arbitrary step of training, define the change rate of the parameter $\mathbf{\theta}$ as

$$
\frac{\partial\mathbf{\theta}}{\partial t} = -\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}) =
-\big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top \nabla_{f_{\mathbf{\theta}}(\mathbf{x})}\mathcal{L}(\mathbf{\theta})
$$

The network Jacobian for any input $\mathbf{x}$ evolves as

$$
\begin{align*}
    \frac{d f_{\mathbf{\theta}}(\mathbf{x})}{dt} &=
    \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top \frac{d\mathbf{\theta}}{dt} \\\\
    &= - \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top \nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}) \\\\
    &= - \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top \nabla_{\mathbf{\theta}}\Big(\frac{1}{n} \sum_{i=1}^n \big(y_i - f_{\mathbf{\theta}}(\mathbf{x}_i)\big)^2\Big) && \text{Expand and compute the gradient} \\\\
    &= - \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top \Big(\frac{1}{n} \sum_{i=1}^n \big(y_i - f_{\mathbf{\theta}}(\mathbf{x}_i)\big)\big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x}_i)\big)\Big) \\\\
    &= - \frac{1}{n}\sum_{i=1}^n \big(y_i - f_{\mathbf{\theta}}(\mathbf{x}_i)\big) \underbrace{\big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\big)^\top\big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x}_i)\big)}_{\text{NTK }\kappa(\mathbf{x}, \mathbf{x}')} && \text{Defined } \kappa(\mathbf{x}, \mathbf{x}') \\\\
\end{align*}
$$

Inside $\kappa(\mathbf{x}, \mathbf{x}_i)$, the $\mathbf{x}$ is a continuous input derived from $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})$, the $\mathbf{x}_i$ is a training point.
In computation, $\mathbf{x}_i$ is often denoted as $\mathbf{x}'$ to refer any arbitrary sample point.
So is $\mathbf{x}$ that $\mathbf{x}, \mathbf{x}'\in X$ can take any actual value from the input space, even let $\mathbf{x}=\mathbf{x}'$ be equal in computation.

$\kappa(\mathbf{x}, \mathbf{x}')$ can be thought of as the covariance of the gradient of the network output with respect to the weights.

### Proof of Kernel Tendency by Chain Rule Across Multiple Layers

Here shows that NTK kernel $\kappa(\mathbf{x}, \mathbf{x}')$ is deterministic across stacked layers through gradient chain rule.

Consider a neural network $f_{\mathbf{\theta}}(\mathbf{x})$ parameterized by $\mathbf{\theta}$ and it is very wide $d\rightarrow\infty$, the stacked layers of the network progress as

$$
\begin{align*}
\mathbf{a}^{(0)}(\mathbf{x}, \mathbf{\theta}) &= \mathbf{x} \\\\
\mathbf{z}^{(L+1)}(\mathbf{x}, \mathbf{\theta}) &= \frac{1}{\sqrt{d_L}}W^{(L)}\mathbf{a}^{(L)}(\mathbf{x}, \mathbf{\theta}) + \mathbf{b}^{(L)} \\\\
\mathbf{a}^{(L+1)}(\mathbf{x}, \mathbf{\theta}) &= \sigma(\mathbf{z}^{(L+1)}(\mathbf{x}, \mathbf{\theta}))
\end{align*}
$$

where $W,\mathbf{b}\sim\mathcal{N}(0,1)$ and $\sigma(.)$ is the activation function.
The normalization term $\frac{1}{\sqrt{d_L}}$ is used to prevent NTK divergence in training.
The parameters are $\mathbf{\theta}=[W^{(1)}, W^{(2)}, \cdots, W^{(L)}, \mathbf{b}^{(1)}, \mathbf{b}^{(2)}, \cdots, \mathbf{b}^{(L)}]$.

The first layer has with $d_1$ neurons has NTK

$$
\begin{align*}
    \kappa^{(1)}(\mathbf{x}, \mathbf{x}') &= \frac{1}{d_1}\sum^{d_1}_{i=1}\frac{\partial f_{\theta_i}^{(1)}(\mathbf{x}, \mathbf{\theta})}{\partial{\theta}_i}\cdot\frac{\partial f_{\theta_i}^{(1)}(\mathbf{x}', \mathbf{\theta})}{\partial{\theta}_i} \\\\
    &= E\big(\big(\nabla_{\mathbf{\theta}}f^{(1)}_{\mathbf{\theta}}(\mathbf{x})\big)^\top\big(\nabla_{\mathbf{\theta}}f^{(1)}_{\mathbf{\theta}}(\mathbf{x}')\big)\big)
\end{align*}
$$

The next layer proof is done by mathematical induction, starting from the $1$-st layer, and then the $l$-th layer, and so on.

Here introduces the covariance matrix $\Sigma^{(l)}(\mathbf{x}, \mathbf{x}')$ for the $l$-th layer.
By chain rule, the NTK kernel $\kappa^{(l)}(\mathbf{x}, \mathbf{x}')$ of the $l$-th layers of the network can be computed as

$$
\begin{align*}
    \kappa^{(l)}(\mathbf{x}, \mathbf{x}') &= \dot{\Sigma}^{(l)}(\mathbf{x}, \mathbf{x}') \cdot
    \underbrace{E\Big(\big(\nabla_{\mathbf{\theta}}f^{(l-1)}_{\mathbf{\theta}}(\mathbf{x})\big)^\top\big(\nabla_{\mathbf{\theta}}f^{(l-1)}_{\mathbf{\theta}}(\mathbf{x}')\big)\Big)}_{\text{NTK }\kappa^{(l-1)}(\mathbf{x}, \mathbf{x}')} +
    \Sigma^{(l)}(\mathbf{x}, \mathbf{x}')
\end{align*}
$$

where $\dot{\Sigma}^{(l)}(\mathbf{x}, \mathbf{x}')$ is the Jacobian of the covariance matrix $\Sigma^{(l)}(\mathbf{x}, \mathbf{x}')$.

The covariance matrix $\Sigma^{(l)}(\mathbf{x}, \mathbf{x}')$ adds up by chain rule, hence the NTK kernel is deterministic.

#### Covariance Matrix Computation

Consider the $l$-th layer with non-linear activation function $\sigma(.)$,

$$
\begin{align*}
\mathbf{z}^{(l)}(\mathbf{x}, \mathbf{\theta}) &= W^{(l-1)}\mathbf{a}^{(l-1)}(\mathbf{x}, \mathbf{\theta}) + \mathbf{b}^{(l-1)} \\\\
\mathbf{a}^{(l)}(\mathbf{x}, \mathbf{\theta}) &= \sigma(\mathbf{z}^{(l)}(\mathbf{x}, \mathbf{\theta}))
\end{align*}
$$

Compute the covariance matrices $\Sigma^{(l)}(\mathbf{x}, \mathbf{x}')$ for $\mathbf{z}^{(l)}$ and $\mathbf{a}^{(l)}$ that for linear $\mathbf{z}^{(l)}$, same linear operation can be applied, while for non-linear activation $\mathbf{a}^{(l)}$, covariance expectation is computed.

$$
\begin{align*}
    \Sigma^{(l)}_{\mathbf{z}}(\mathbf{x}, \mathbf{x}') &=
    \sigma_w^2 \Sigma^{(l-1)}_{\mathbf{a}}(\mathbf{x}, \mathbf{x}') + \sigma_{\mathbf{b}}^2 \\\\
    \Sigma^{(l)}_{\mathbf{a}}(\mathbf{x}, \mathbf{x}') &=
    E\Big(\big(\sigma(\mathbf{z}^{(l)}(\mathbf{x}, \mathbf{\theta}))\big)^{\top}\big(\sigma(\mathbf{z}^{(l)}(\mathbf{x}', \mathbf{\theta}))\big)\Big)
\end{align*}
$$

where for $\Sigma^{(l)}_{\mathbf{a}}(\mathbf{x}, \mathbf{x}')$, assume the input $\mathbf{z}^{(l)}(\mathbf{x}, \mathbf{\theta})$ and $\mathbf{z}^{(l)}(\mathbf{x}', \mathbf{\theta})$ follow joint Gaussian distribution where mean is $\mathbf{0}$ and covariance is $\Sigma_{\mathbf{x}}^{(l)}(\mathbf{x}, \mathbf{x}')$, i.e.,

$$
\Sigma^{(l)}_{\mathbf{a}}(\mathbf{x}, \mathbf{x}') =
\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \sigma(\mathbf{u}) \sigma(\mathbf{v})
\cdot \mathcal{N}\Big(
    \begin{bmatrix} \mathbf{u} \\\\ \mathbf{v}
    \end{bmatrix};
    \begin{bmatrix} \mathbf{0} \\\\ \mathbf{0}
    \end{bmatrix},
    \begin{bmatrix}\Sigma_{\mathbf{x}}^{(l)}(\mathbf{x}, \mathbf{x}) & \Sigma_{\mathbf{x}}^{(l)}(\mathbf{x}, \mathbf{x}') \\\\
    \Sigma_{\mathbf{x}}^{(l)}(\mathbf{x}', \mathbf{x}) & \Sigma_{\mathbf{x}}^{(l)}(\mathbf{x}', \mathbf{x}')
    \end{bmatrix}
\Big) d\mathbf{u} d\mathbf{v}
$$

### Proof of Jacobian Stability by Existed Infinitesimal Hessian Matrix

Here shows that the Jacobian at initialization can be approximated by $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}}(\mathbf{x})\approx\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$, i.e., as iteration step $\Delta t$ increases, there is $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_t}(\mathbf{x})\approx\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$.

$$
f_{\mathbf{\theta}_t}(\mathbf{x}) \approx f_{\mathbf{\theta}_0}(\mathbf{x}) + \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})\big)^{\top} (\mathbf{\theta}_t - \mathbf{\theta}_0)
$$

Let $\mathbf{\theta}_{t+1}-\mathbf{\theta}_{t} = -\eta\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta}_t)$ be a parameter update step,
then here shows proof of Jacobian stability, that at an iteration step, for Jacobian is governed by Hessian matrix, that $||\text{Hessian}(.)||\rightarrow 0$ as $d\rightarrow\infty$, and thus the Jacobian is stable, i.e.,

$$
||\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_t}(\mathbf{x})-\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})|| \leq
||\text{Hessian}(f_{\mathbf{\theta}_0}(\mathbf{x}))|| \cdot ||\mathbf{\theta}_t - \mathbf{\theta}_0|| \approx 0
$$

Recall the definition of Hessian matrix (the second-order partial derivatives of a function),

$$
\begin{align*}
    \text{Hessian}\big(f_{\mathbf{\theta}}(\mathbf{x})\big) &= \frac{d^2 f_{\mathbf{\theta}}(\mathbf{x})}{dt^2} \\\\
\end{align*}
$$

where $\frac{d f_{\mathbf{\theta}}(\mathbf{x})}{dt} = - \kappa(\mathbf{x}, \mathbf{x}') \big|\big|\mathbf{y} - f_{\mathbf{\theta}}(\mathbf{x})\big|\big|$.

Here $\big|\big|\mathbf{y} - f_{\mathbf{\theta}}(\mathbf{x})\big|\big|$ is a scalar value given a static training dataset $\mathcal{D}$ and optimized parameters $\mathbf{\theta}$ (which is also static at a convergence iteration),
and $\kappa(\mathbf{x}, \mathbf{x}')$ is  mainly concerned with the covariance matrices. $\Sigma(\mathbf{x}, \mathbf{x}')$ and $\dot{\Sigma}(\mathbf{x}', \mathbf{x}')$.
Remove the scalar value $\big|\big|\mathbf{y} - f_{\mathbf{\theta}}(\mathbf{x})\big|\big|$, the Hessian matrix is proportional to the NTK kernel $\kappa^2(\mathbf{x}, \mathbf{x}')$.

$$
\frac{d^2 f_{\mathbf{\theta}}(\mathbf{x})}{dt^2} \propto \kappa^{2}(\mathbf{x}, \mathbf{x}')
$$

where $\kappa^{(l)}(\mathbf{x}, \mathbf{x}') = \dot{\Sigma}^{(l)}(\mathbf{x}, \mathbf{x}') \cdot \kappa^{(l-1)}(\mathbf{x}, \mathbf{x}') + \Sigma^{(l)}(\mathbf{x}, \mathbf{x}')$.

Remember that weights are initialized from Gaussian distribution with a total variance of $1$ as well as in training process, i.e.,

$$
E\big(W^{(l)}_{ij}W^{(l)}_{ik}\big) = \frac{\sigma_w^2}{d_{l-1}}\delta_{jk}, \quad
\sigma_w^2 = \mathcal{O}(1)
$$

where $\delta_{jk}\sim\mathcal{O}(1)$ is the Kronecker delta $\delta_{jk}=\begin{cases}1 & j=k \\\\ 0 & j\neq k\end{cases}$,
that specifies each element of the weight matrix multiplcation has the $\mathcal{O}(\frac{1}{d_{l-1}})$ variance.

Proof: the sum of $d$ random variables with individual variance $\sigma_i^2$ has a variance of $\sum_{i=1}^d\sigma_i^2$.
To contain the total variance $1=\sum_{i=1}^d\sigma_i^2$, each random variable variance should be $\sigma_i^2=\frac{1}{d}$, i.e., be initialized as $\sigma_i=\frac{1}{\sqrt{d}}$.

Key insight is here: as a network goes very wide $d\rightarrow\infty$, each individual weight update $\delta_{ij} W^{(l)}_{ij}$ is $\mathcal{O}(\frac{1}{\sqrt{d_{l-1}}})$ that is already very small, let alone $\kappa^2(\mathbf{x}, \mathbf{x}')$ is the squared operation with respect to $\Sigma^{(l)}(\mathbf{x}, \mathbf{x}')$ and $\dot{\Sigma}^{(l)}(\mathbf{x}, \mathbf{x}')$.
As a result, the Hessian matrix is dominated by $\kappa^2(\mathbf{x}, \mathbf{x}')$ and is infinitesimal, i.e., $\frac{d^2 f_{\mathbf{\theta}}(\mathbf{x})}{dt^2}\rightarrow 0$.

For the Hessian matrix is infinitesimal, the Jacobian is stable/invariant,
and for Jacobian is stable/invariant, the Jacobian at initialization can be considered invariant given a iteration step $\Delta t$, i.e., $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_t}(\mathbf{x})\approx\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$.

### Proof of NTK That A Wide Network Is Essentially A Linear Network

In the above proof, it has already established that

$$
f_{\mathbf{\theta}_t}(\mathbf{x}) \approx f_{\mathbf{\theta}_0}(\mathbf{x}) + \big(\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})\big)^{\top} (\mathbf{\theta}_t - \mathbf{\theta}_0)
$$

where $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_t}(\mathbf{x})\approx\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})$ is an invariant gradient vector.

Now consider training a purely linear model and expand it by Taylor series (the first-order term in linear expansion always holds).
Start from any arbitrary iteration, there is

$$
g^{\text{linear}}_t(\mathbf{x}) \equiv g^{\text{linear}}_{\mathbf{\theta}_0}(\mathbf{x}) + \big(\nabla_{\mathbf{\theta}}g^{\text{linear}}_{\mathbf{\theta}_0}(\mathbf{x})\big)^{\top} (\mathbf{\theta}_t - \mathbf{\theta}_0)
$$

If $f_{\mathbf{\theta}_t}(\mathbf{x})$ and $g^{\text{linear}}_t(\mathbf{x})$ see identical outputs given same input $\mathbf{x}$, and the initializations are the same, the update gradient vector must be the same, i.e., $\nabla_{\mathbf{\theta}}f_{\mathbf{\theta}_0}(\mathbf{x})=\nabla_{\mathbf{\theta}}g^{\text{linear}}_{\mathbf{\theta}_0}(\mathbf{x})$.

It can be said that when $d\rightarrow\infty$, $f_{\mathbf{\theta}_t}(\mathbf{x})$ is equivalent to $g^{\text{linear}}_t(\mathbf{x})$.
