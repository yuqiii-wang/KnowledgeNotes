# Boosting

Boosting is an ensemble learning mechanism that strategically combines multiple "weak learners" to create a single, highly accurate "strong learner" by an additive manner.

Let $F(x)$ be the ensemble strong learner.
Set $T$ sequential stages to train $F(x)$.
Each time train a weak learner $f_t(x)$.

This new weak learner $f_t(x)$ sees what samples are mis-predicted, assigns heavy weights on those failure samples in training.
The weak learner itself has a weight indicative of how good it is in prediction.

Finally, all weak learners are "combined" by an additive manner as a single monolith strong learner $F(x)=\sum_t f_t(x)$.

Boosting enhances weak learners as specialists to handle edge cases, hence

* outperform a regular monolith model that may ignore edge cases
* advantageous for imbalanced data
* sensitive to noises

Boosting is ensemble learning mechanism NOT an actual learning algo.
There needs a base learner such as a decision tree or SVM.

## AdaBoost

AdaBoost (short for Adaptive Boosting) is a naive implementation of the boosting philosophy "Can a set of weak learners create a single strong learner?" (proposed in 1990).

Suppose now to teach $F$ to predict values of the form $\hat{y}=F(x)$ by minimizing the error $\mathcal{L}(y,F(x))$ for a size of $n$ samples indexed by $i$.

If the algorithm has $T$ training stages (also a total of $T$ weak learners), at each stage $t$ there is a trained weak learner $f_t$.
Define at the $t$-th stage a weak learner produces an output hypothesis $h_t(x_i)$ for each sample in the training set.

Let $\alpha_t$ be weights to a weak learner output $h_t(x)$, there is

$$
F_t(x_i)=y_i=F_{t-1}(x_i)+\alpha_t h_t(x_i)
$$

$F_t$ is the boosted model that has been built up to the previous stage of training $F_{t-1}$.
Expand the recursion, there is

$$
\begin{align*}
F_t(x_i) &= F_{t-1}(x_i)+\alpha_t h_{t}(x_i) \\\\
 &= F_{t-2}(x_i)+\alpha_{t-1} h_{t-1}(x_i)+\alpha_t h_{t}(x_i) \\\\
 &= F_{t-3}(x_i)+\alpha_{t-2} h_{t-2}(x_i)+\alpha_{t-1} h_{t-1}(x_i)+\alpha_t h_{t}(x_i) \\\\
 &= ... 
\end{align*}
$$

The minimization objective is the sum of each stage $t$-th training loss.
This is equal to minimizing each stage loss, so that the sum of all stages' is minimized.

$$
\begin{align*}
\text{Total Loss}&& \min_{y,F(x)} \mathcal{L} &=\sum_{i=1}^n \mathcal{L}\Big(y_i,\sum_{t=1}^T F_t(x_i)\Big) \\\\
\text{Each Stage Loss}&& \min_{h(x),F(x)} \mathcal{L}_t &=\sum_{i=1}^n \mathcal{L}\Big(y_i, F_{t-1}(x_i)+\alpha_t h_t(x_i)\Big) \\\\
&& &=\sum_{i=1}^n \mathcal{L}\Big(y_i,\sum_{\tau=1}^{t-1} \alpha_{\tau}f_{\tau}(x_i)+\alpha_t h_t(x_i)\Big)
\end{align*}
$$

where $f_t(x)=\alpha_t h_t(x)$ is the weak learner that is being considered for addition to the final model.
The so-called "strong learner" is $F_t(x)=\sum_{\tau}\alpha_{\tau} h_{\tau}(x)$.

As training progresses **samples** and **weak learners** are adaptively weighted.

### AdaBoost Adaptive Weight Derivation by A Binary Classification Example

Assume the task is to learn a binary classification $y_i\in Y=\{-1,+1\}^n$ from a dataset $X\sim D_{t}$.
The $D_{t}$ represents samples $\{(x_i,y_i)\}$ at the $t$-th training stage are adaptively weighted based on previous stage dataset $D_{t-1}$.

In the beginning set sample weights as all equal $w_{i,0}=\frac{1}{n}$.

Each time the estimate is $h_t(x_i)\in\{-1,+1\}$.

Assumed using exponential as loss.

$$
\mathcal{L}(F)=\text{exp}(-Y F(x))
$$

#### Sample Weight Adaptation

To reflect each next new weak learner should learn previous training stage failed samples, the sample weight adaption should be proportional to the previous stage exponential loss.

$$
\begin{align*}
D_t(x) &\propto D_{t-1}(x) \space\space \text{exp}\Big(y_i \alpha_t F_{t-1}(x_i)\Big) \\\\
&= D_{t-1}(x)\space\cdot
\begin{cases}
  e^{-\alpha_t} & Y=F_{t-1}(x) \\\\
  e^{\alpha_t} & Y\ne F_{t-1}(x)
\end{cases} \\\\
&\propto
D_{t-1}(x)\space\cdot
\begin{cases}
  1 & Y=F_{t-1}(x) \\\\
  e^{2\alpha_t} & Y\ne F_{t-1}(x)
\end{cases}
\end{align*}
$$

When prediction by $F_{t-1}(x)$ is correct, samples before and after the $t$-th training stage have equal weights $w_{t,i}=1$;
otherwise $D_t(x)$ sees weight multiplication by $w_{t,i}=e^{2\alpha_t}$.

The effect of sample weight distribution sees $w_{t,i}\sim D_t$ applied on loss with respect to individual samples.

$$
\mathcal{L}_t=\sum^n_{i=1} w_{t,i} \cdot\mathcal{L}_{t,i}(y_i, \hat{y}_i)
$$

Heavy weights see the sample loss $w_{t,i} \cdot\mathcal{L}_{t,i}(y_i, \hat{y}_i)$ getting amplified.

#### Learner Weight Adaptation

The $t$-th stage weak learner weight $\alpha_t$ choice derives from $\alpha_t=\argmin_{\alpha}\mathcal{L}_t(F)$ that

$$
\begin{align*}
\mathcal{L}_t(F) &=\text{exp}(-Y F_t(x)) \\\\
&= \text{E}_{X\sim D_{t-1}} \exp\Big(-Y\big(F_{t-1}(x)+\alpha_{t-1} h_{t-1}(x)\big)\Big) \\\\
&= \text{E}_{X\sim D_{t-1}} \Big[\exp\Big(-YF_{t-1}(x)\Big)\cdot\text{exp}\Big(-Y\alpha_{t-1} h_{t-1}(x)\Big)\Big] \\\\
&= \text{E}_{X\sim D_t} \space\text{exp}\Big(Y\alpha_{t-1} h_{t-1}(x)\Big) &&\quad \text{for }D_{t-1} \rightarrow D_t \text{ samples are adaptively weighted} \\\\
&&&\quad D_t(x) \propto D_{t-1}(x) \space\space \text{exp}\Big(y_i \alpha_t F_{t-1}(x_i)\Big) \\\\
&= \text{E}_{X\sim D_t} \Big[e^{-\alpha_t}\mathbb{1}\big(Y=F_t(x)\big)+e^{\alpha_t}\mathbb{1}\big(Y\ne F_t(x)\big)\Big] &&\quad 0\le\epsilon_t\le 1 \text{ is the normalized mean mis-classification of } n \text{ samples}\\\\
&= e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t}\epsilon_t \\\\
\end{align*}
$$

Find the argument $\alpha_t$ that leads to minimal $\mathcal{L}_t(F)$ can be computed by setting the derivative to zero.

$$
\begin{align*}
&& \frac{\partial \mathcal{L}_t(F)}{\partial \alpha_t}=0&=-e^{-\alpha_t}(1-\epsilon_t) + e^{\alpha_t}\epsilon_t \\\\
\Rightarrow && e^{-\alpha_t}(1-\epsilon_t) &= e^{\alpha_t}\epsilon_t \\\\
\Rightarrow && e^{-2\alpha_t}(1-\epsilon_t) &= \epsilon_t \\\\
\Rightarrow && \ln\Big(e^{-2\alpha_t}(1-\epsilon_t)\Big) &= \ln\epsilon_t \\\\
\Rightarrow && -2\alpha_t\ln(1-\epsilon_t) &= \ln\epsilon_t \\\\
\Rightarrow && \alpha_t &= \frac{1}{2}\ln\Big(\frac{1-\epsilon_t}{\epsilon_t}\Big)
\end{align*}
$$

The learner weight is a monotonic function of error rate $\epsilon_t$ that the expression $\alpha_t=\frac{1}{2}\ln\Big(\frac{1-\epsilon_t}{\epsilon_t}\Big)$ says

* Small error rate $\epsilon_t\approx 0$ gives $\alpha_t\rightarrow +\infty$ that this weak learner $f_t(x)=\alpha_t h_t(x)$ has significant positive importance
* Large error rate $\epsilon_t\approx 1$ gives $\alpha_t\rightarrow -\infty$ that this weak learner $f_t(x)=\alpha_t h_t(x)$ has significant negative importance

## Gradient Boosting

To learn $\{(x_i, y_i)\}$, different from AdaBoost where $h_t(x)$ learns from the binary classification problem that the progress adaptively updates sample and learner weights,
gradient boosting training computes *pseudo residual* $r_{t,i}$

$$
r_{t,i}=y_i-F_t(x_i)
$$

$r_{t,i}$ is termed pseudo residual for that it is an approximation to real residual $r_i=\hat{y}_i-y_i$ that $\hat{y}_i$ should be output from the final model $F(x)$.
Instead, $r_{t,i}$ is an iterative approach to approximate the real $r_i$.

Consider the 1st order Taylor expansion of a regular function ($a$ is a stationary point)

$$
\begin{align*}
f(x)&\approx f(a)+f'(a)\cdot(x-a)
\end{align*}
$$

Accordingly, the loss by the 1st order Taylor expansion is

$$
\begin{align*}
\mathcal{L}_t\Big(y,F_t(x)\Big) &\approx
\mathcal{L}\Big(y,F_{t-1}(x)\Big)+\frac{\partial\mathcal{L}(y,F(x))}{\partial F(x)}\bigg|_{F(x)=F_{t-1}(x)}\cdot\bigg(F_{t}(x)-F_{t-1}(x)\bigg) \\\\
&= \mathcal{L}\Big(y,F_{t-1}(x)\Big)+\frac{\partial\mathcal{L}(y,F(x))}{\partial F(x)}\bigg|_{F(x)=F_{t-1}(x)}\cdot f_t(x) \\\\
\end{align*}
$$

To minimize $\mathcal{L}_t\Big(y,F_t(x)\Big)$, the weak learner proposed $h_t(x)$ should approach the residual such that $h_t(x)=\argmin\sum^n_i \big(r_{t,i}-h_t(x_i)\big)^2$.
This is different from AdaBoost $F(x)=\sum_t f_t(x)=\sum_t \alpha_t h_t(x)$ that sees $h_t(x_i)\in\{-1,+1\}$ as the estimate of each time the trained weak learner.

$\mathcal{L}\Big(y,F_{t-1}(x)\Big)$ is constant and removed in optimization.

To update model, there is

$$
F_t(x)=F_{t-1}(x)+\eta h_t(x)
$$

where $\eta$ is learning rate.

Interestingly, if the loss is Mean Square Error (MSE), the pseudo residual is exactly the same as MSE residual.

$$
\begin{align*}
\frac{\partial\mathcal{L}(y,F(x_i))}{\partial F(x_i)}&=
\frac{\partial}{\partial F(x_i)}\bigg(\frac{1}{2}\big(y_i-F(x_i)\big)^2\bigg) \\\\
&= -\big(y_i-F(x_i)\big)
\end{align*}
$$

This gives that the gradient is exactly the same as residual.

If used Huber Loss, there is

$$
h_t(x)=-\frac{\partial\mathcal{L}(y,F(x_i))}{\partial F(x_i)}\bigg|_{F(x)=F_{t-1}(x)}=
\begin{cases}
  y-F(x) & |y-F(x)|\ge\delta \\\\
  \delta \cdot \text{sgn} \big(y-F(x)\big) & |y-F(x)|<\delta \\\\
\end{cases}
$$

### GBDT (Gradient Boosting Decision Tree)

Reference:
https://aandds.com/blog/ensemble-gbdt.html

GBDT (Gradient Boosting Decision Tree) implements $F(x)$ by decision tree.
For $T$ training stages, there are $T$ decision trees.

In GBDT, terminal regions refer to decision tree leaf node input ranges.
Such terminal regions stop further tree branch forking.

Assume there are $T$ decision trees that each got trained at a time $t$, and each tree got $J_t$ terminal regions/leaf nodes.
The terminal region is denoted as $R_{t,j}$ where $0<j_t\le J_t$.

For every $R_{t,j}$, compute the optimal pseudo residual of each $\gamma_{t,j}$ that in sum to approximate the total residual $\sum_j\gamma_{t,j}\rightarrow r_{t}$.

$$
\text{For } j = 1,2,...,J_t:\qquad
\gamma_{t,j}^* = \argmin_{\gamma_{t,j}} \sum_{x_i\in R_{t,j}} \mathcal{L}\Big(y_i, F_{t-1}(x_i)+ \gamma_{t,j}\Big)
$$

Provided the optimal $\gamma_{t,j}$ for each terminal region that aims to approximate a fraction of pseudo residual,the update becomes

$$
F_t(x)=F_{t-1}(x)+\sum^{J_t}_{j=1}\gamma_{t,j} I\big(x_i\in R_{t,j}\big)
$$

where $I(.)$ is an indicator function output $1$ when argument condition is true; $0$ otherwise, i.e.,
$I\big(x_i\in R_{t,j}\big)=\begin{cases}1&x_i\in R_{t,j}\\0&x_i\notin R_{t,j}\end{cases}$.

#### Why Numeric Differentiation

In optimization, for a tree structure the automatic differential is not applicable, the implemented differential is numerical differential directly on the loss.

Automatic differentiation (AD) breaks down a function into a sequence of elementary, differentiable operations (like $+$, $\times$, $\sin$, $\exp$) applying the chain rule.
A decision tree, however, is not a single, smooth mathematical function. It is a set of hierarchical, discrete rules, e.g., contained lots of $\text{if}/\text{else}$ branches.

#### GBDT Training

1. Assumed an fixed tree structure $F_t(x)$
2. Find the optimal leaf output $\gamma_{t,j}^* = \argmin_{\gamma_{t,j}} \sum_{x_i\in R_{t,j}} \mathcal{L}\Big(y_i, F_{t-1}(x_i)+ \gamma_{t,j}\Big)$
3. For the node $j$, compare parent loss $\mathcal{L}_{t,j,\text{parent}}$ vs. if split to two children $\mathcal{L}_{t,j,\text{leftChild}}+\mathcal{L}_{t,j,\text{rightChild}}$
4. The tree structure is updated

### XGBoost (eXtreme Gradient Boosting)

XGBoost is built on top of GBDT with enhancements such as

* Second order Taylor Expansion for better convergence
* Regularization $\mathcal{\Omega}_{\tau}\big(F_t\big)$ for better generalization
* Data storage by blocks to allow computation parallelization

Considered an added regularization term, the optimization objective is

$$
\min_{y,F(x)} \mathcal{L} = \sum_{i=1}^n \mathcal{L}_t\Big(y_i, F_t(x_i)\Big)+
\sum_{\tau=1}^{t-1} \mathcal{\Omega}_{\tau}\big(F_t\big) + \mathcal{\Omega}_{t}\big(F_t\big)
$$

where regularization term is

$$
\mathcal{\Omega}_{t}\big(F_t\big)=
\lambda J_t + \frac{1}{2}\lambda\sum^{J_t}_{j=1}\gamma_{t,j}^2
$$

where

* $\lambda$ is a regularization coefficient to control the significance of the regularization effect.
* $\lambda J_t$ controls the complexity of a tree that the num of leaf nodes should NOT be too many.
* $\frac{1}{2}\lambda\sum^{J_t}_{j=1}\gamma_j^2$ balances individual leaf nodes' importance that neither does a single node have heavy nor light weight.

#### Derive Optimal Leaf Output $\gamma_j^*$ in Each Tree Training

Consider the 2nd order Taylor expansion of a regular function ($a$ is a stationary point)

$$
\begin{align*}
f(x)&\approx f(a)+f'(a)\cdot(x-a)+\frac{1}{2}f''(a)\cdot(x-a)^2
\end{align*}
$$

Accordingly, the loss with regularization is  

$$
\begin{align*}
\mathcal{L}_t\Big(y,F_t(x)\Big) &\approx \mathcal{L}\Big(y,F_{t-1}(x)\Big)+
\frac{\partial\mathcal{L}(y,F(x))}{\partial F(x)}\bigg|_{F(x)=F_{t-1}(x)}\cdot f_t(x)+
\frac{1}{2}\frac{\partial^2\mathcal{L}(y,F(x))}{\partial F^2(x)}\bigg|_{F(x)=F_{t-1}(x)}\cdot f^2_t(x) +
\sum_{\tau=1}^{t-1} \mathcal{\Omega}_{\tau}\big(F_t\big) + \mathcal{\Omega}_{t}\big(F_t\big) \\\\
&= \mathcal{L}\Big(y,F_{t-1}(x)\Big)+
g_t(x)\cdot f_t(x)+\frac{1}{2}h_t(x)\cdot f^2_t(x)+
\sum_{\tau=1}^{t-1} \mathcal{\Omega}_{\tau}\big(F_t\big) + \mathcal{\Omega}_{t}\big(F_t\big)
 & \text{Gradient and Hessian represented as } g_t(x) \text{ and } h_t(x)
\end{align*}
$$

where $\mathcal{L}\Big(y,F_{t-1}(x)\Big)$ and $\sum_{\tau=1}^{t-1} \mathcal{\Omega}_{\tau}\big(F_t\big)$ are constant and can be removed in optimization.
As a result, the real target is to minimize

$$
\tilde{\mathcal{L}}_t\Big(y,F_t(x)\Big)=
g_t(x)\cdot f_t(x)+\frac{1}{2}h_t(x)\cdot f^2_t(x) + \mathcal{\Omega}_{t}\big(F_t\big)
$$

Now, the goal here is to find the optimal output $\gamma_{t,j}$ for each leaf, assuming the structure of the tree ($J_t$ and the instance/terminal region sets $\text{For } j = 1,2,...,J_t:\space R_{t,j}$) is fixed.

Rewrite the Objective Function in Terms of Leaves

$$
\begin{align*}
\tilde{\mathcal{L}}_t\Big(y,F_t(x)\Big)&=
g_t(x)\cdot f_t(x)+\frac{1}{2}h_t(x)\cdot f^2_t(x) + \mathcal{\Omega}_{t}\big(F_t\big) \\\\
&= \sum^{J_t}_{j=1}\sum_{x_i\in R_{t,j}}\bigg(g_t(x_i)\cdot \gamma_{t,j}+\frac{1}{2}h_t(x_i)\cdot \gamma_{t,j}^2 \bigg)+
\lambda J_t + \frac{1}{2}\lambda\sum^{J_t}_{j=1}\gamma_{t,j}^2 \\\\
&= \sum_{j=1}^{J_t} \left[ \left(\sum_{x_i\in R_{t,j}} g_t(x_i)\right) \gamma_{t,j} + \frac{1}{2}\left(\sum_{x_i\in R_{t,j}} h_t(x_i)\right) \gamma_{t,j}^2 \right] + \lambda J_t + \frac{1}{2}\lambda \sum_{j=1}^{J_t} \gamma_{t,j}^2 \\\\
&= \sum_{j=1}^{J_t} \left[ \left(\sum_{x_i\in R_{t,j}} g_t(x_i)\right) \gamma_{t,j} + \frac{1}{2}\left(\sum_{x_i\in R_{t,j}} h_t(x_i)+\lambda\right) \gamma_{t,j}^2 \right] + \lambda J_t \\\\
\end{align*}
$$

where, provided a sample indexed by $i$ flown through a tree reaching leaf node indexed by $j$ it shall get output, $\left(\sum_{x_i\in R_{t,j}} (.)\right) \gamma_{t,j}$ means all samples if reached $j$ shall get $\gamma_{t,j}$ which is a fraction of total residual $\sum^{J_t}_{j=1}\gamma_{t,j} I\big(x_i\in R_{t,j}\big)$.

Rewrite $\tilde{\mathcal{L}}_t\Big(y,F_t(x)\Big)$ in matrix format, there is $\mathcal{L}_{t,j}(\gamma_{t,j}) = G_j \gamma_{t,j} + \frac{1}{2}(H_j + \lambda) \gamma_{t,j}^2$

Find the optimal $\gamma_j^*$

$$
\begin{align*}
&& \frac{\partial \tilde{\mathcal{L}}_t\Big(y,F_t(x)\Big)}{\partial \gamma_{t,j}} &=G_j + (H_j + \lambda)\gamma_{t,j}^* = 0 \\\\
\Rightarrow && \gamma_{t,j}^* &= - \frac{G_j}{H_j + \lambda}
\end{align*}
$$

#### Tree Branch Fork by Gain

Substitute $\gamma_{t,j}^*=-\frac{G_j}{H_j + \lambda}$ back to the loss function, the minimal loss is

$$
\begin{align*}
\mathcal{L}_{t,j}\left(\gamma_{t,j}^*=-\frac{G_j}{H_j + \lambda}\right) &= G_j \gamma_{t,j} + \frac{1}{2}(H_j + \lambda) \gamma_{t,j}^2 \\\\
&= -\frac{G_R^2}{H_R + \lambda}
\end{align*}
$$

The negative of the node loss $-\mathcal{L}_{t,j}$ is named *score*.

The decision of a tree branch forking is based on *gain*,
which is the summed left + right children vs. parent score.

$$
\begin{align*}
\text{Gain} &= \underbrace{\frac{1}{2} \frac{G_L^2}{H_L + \lambda}}_{\text{Left Child Score}} + \underbrace{\frac{1}{2} \frac{G_R^2}{H_R + \lambda}}_{\text{Right Child Score}} - \underbrace{\frac{1}{2} \frac{G_I^2}{H_I + \lambda}}_{\text{Parent Score}}
\end{align*}
$$

If $\text{Gain}>0$, split the leaf node; otherwise, keep the leaf node as terminal.
