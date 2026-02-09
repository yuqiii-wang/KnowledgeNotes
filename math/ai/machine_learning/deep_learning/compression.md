# Model Compression: Pruning, Distillation and Quantization

This article shows three popular compression methods:

* Pruning
* Quantization
* Distillation

## Pruning

Pruning permanently drops certain weights.

In pytorch, there is `import torch.nn.utils.prune as prune`.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

parameters_to_prune = (
            (model.conv1, 'weight'),
            (model.conv2, 'weight'),
            (model.fc1, 'weight'),
            (model.fc2, 'weight'),
            (model.fc3, 'weight'))

## prune all layer parameters removing 20% of which by L1 unstructured pruning method (individual neuron removals by L1)
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
```

### Best Practices and Lottery Ticket Hypothesis

*The Lottery Ticket Hypothesis: Training Pruned Neural Networks* proves that a subnet contains the same inference capability as a its parent large network.

*lottery ticket hypothesis*: dense, randomly-initialized, feed-forward networks contain subnetworks (winning tickets) that, when trained in isolation, reach test accuracy comparable to the original network in a similar number of iterations.

Good parameter initialization achieves more than $90\%$ compression rate.
Finding good parameter initialization is particularly important.

### Unstructured vs Structured Pruning

* Unstructured Pruning: prune individual parameters
* Structured Pruning: removes entire structures of parameters (e.g., the whole layer of parameters)

### Local vs Global Pruning

* Local: prune as per layer
* Global: prune as multiple layers, or all layers

Model's width per layer represents content of input;
model's depth represents non-linearity capability of transforming input.

Figure below shows that comprehensive pruning (width + depth) is better rather than just pruning on single layer neurons.

<div style="display: flex; justify-content: center;">
      <img src="imgs/pruning_effectiveness.png" width="30%" height="30%" alt="pruning_effectiveness" />
</div>
</br>

### Pruning Methods: By L1, Ln Norms

The key idea is that, norm of a matrix $||W||_p$ describes the average info contained in one neuron.
Insignificant neurons should see low activation energy (low input value), hence the weights should have small $|w_i|_p$.

A small $|w_i|_p$ multiplied with input $x_i$ gives a small value, that when passed to activation function, the activation outputs are almost certain for its input is almost always zeros.

For example, illustrated in the figure below, for a pruning rate of $50\%$ (remove half of parameters) and by $\mathcal{J}_1$ norm of $|W|$, the lowest $|w_i|$ are set to zeros.

<div style="display: flex; justify-content: center;">
      <img src="imgs/pruning_by_weight_magnitude.png" width="50%" height="20%" alt="pruning_by_weight_magnitude" />
</div>
</br>

The above $\mathcal{J}_1$ norm removal can be done in Pytorch such as below.
The original weight $W$ is renamed by prune as `weight_orig`, and the corresponding binary mask is `weight_mask` that sets lowest element to zeros, and retains full pass of original weights element-wise multiplying by $1$.

$$
W \odot M_{mask} = W_{pruned}, \qquad
\text{where }
M_{mask} = \begin{bmatrix}
    1 & 0 & 0 & 0 \\\\
    1 & 1 & 0 & 1 \\\\
    1 & 0 & 0 & 1 \\\\
\end{bmatrix}
$$

where $\odot$ is an element-wise multiplication operator.

```python
m = prune.l1_unstructured(nn.Linear(3, 4), 'weight', amount=0.5)
m.state_dict().keys()
# print
# odict_keys(['bias', 'weight_orig', 'weight_mask'])
```

* $\mathcal{J}_1$ Norm: remove the lowest $|w_i|$:

PyTorch implementation (remove the specified `amount` of (currently un-pruned) units with the lowest L1-norm) reference:
https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html#torch.nn.utils.prune.l1_unstructured

```python
torch.nn.utils.prune.l1_unstructured(module, name, amount, importance_scores=None)
```

* $\mathcal{J}_2$ Norm: remove lowest $w_i^2$

PyTorch implementation (remove the specified `amount` of (currently un-pruned) channels along the specified `dim` with the lowest L`n=2`-norm) reference:
https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html

```python
torch.nn.utils.prune.ln_structured(module, name, amount, n, dim, importance_scores=None)
```

### Whole Model Pruning

For efficient memory storage purposes, large models need pruning.
However, the unstructured pruning (set small $w_i$ to zeros) cannot help for in memory data/tensors need to be aligned in storage, hence all tensors should have the same size.

To reduce memory consumption, need to prune on a whole layer at least.

## Distillation

Knowledge distillation or model distillation is the process of transferring knowledge from a large model (teacher) to a smaller one (student).

During the distillation process, the teacher model generates a set of soft targets, which are essentially probability distributions over the possible next tokens in a sentence.
The student model is then trained to mimic these soft targets, rather than the actual outputs of the teacher model.

* Offline distillation: the pre-trained teacher model remains frozen while the student model is trained. 

* Online distillation: both the teacher and students models are trained simultaneously.

* Self-distillation: the same model acts as both teacher as well as student (self-learning) in training, and allows minor prediction accuracy degradation. The produced student model should be much smaller saving memory.

### Formulation

The learning to mimic the soft targets means that, for example, to learn from BERT-large, there should see $p_{\theta_{\text{BERT-distilled}}}(w_t | \mathbf{w}_{1:t-1}) = p_{\theta_{\text{BERT-large}}}(w_t | \mathbf{w}_{1:t-1})$ (for shorthand notations, denote $p_{\theta_{t}}$ as the teacher model output probability distribution, $p_{\theta_{s}}$ as the student model's, in which $\theta_t$ and $\theta_s$ are models' parameters), where $p_{\theta_s}(w_t)$ is a discrete probability distribution of $30522$ tokens, rather than producing the exactly the same token such that $p_{\theta_s}\big(w_t = \argmax(p_{\theta_{t}}(w_t))\big)=1$.

Since the knowledge transfer result is evaluated based on two probability distributions, here introduces *Kullback-Leibler divergence* $D_{KL}(P || Q)$, that it measures how a student model output probability distribution $Q$ is different from teacher model output probability distribution $P$. 

$$
D_{KL}(P || Q) =
\sum_{x \in X} P(x) \log \Big( \frac{P(x)}{Q(x)} \Big)
$$

A forward KL divergence is defined $D_{KL}(P || Q)$, and reverse KL divergence is $D_{KL}(Q || P)$.
In knowledge transfer/model distillation using a small model to learn from a large model such that $\text{rank}(\theta_Q) < \text{rank}(\theta_P)$, forward KL divergence suffers from "spreading flat" treating the teacher model output $P$ as weights to $\log \frac{P(x)}{Q(x)}$, so that the learnt student's model output $Q$ is "flat".

Reverse KL divergence $D_{KL}(Q || P)=\sum_{x \in X} Q(x) \log \frac{Q(x)}{P(x)}$ is better served as the loss in knowledge distillation, where student model output $Q$ is used as weights to $\log \frac{Q(x)}{P(x)}$ that minimizes the difference between $Q$ and $P$ distributions.
This enables the student learning the most prominent features of the teacher.

To formulate the optimization problem for LLM knowledge distillation, propose the below objective, where $\mathbf{x}$ is the input prompt.
The objective is for the student to best produce the exactly the same token sequence distribution of the teacher's, and this is measured by minimizing the expectation of $\log \frac{Q_{\theta_s}(\mathbf{w} | \mathbf{x})}{P_{\theta_t}(\mathbf{w} | \mathbf{x})}$.
The expectation is applied with weights by $Q_{\theta_s}(\mathbf{w} | \mathbf{x})$ in reverse $D_{KL}$.

$$
\begin{align*}
\theta_s^* &= \argmin_{\theta_s} Q_{\theta_s}(\mathbf{w} | \mathbf{x}) \log \frac{Q_{\theta_s}(\mathbf{w} | \mathbf{x})}{P_{\theta_t}(\mathbf{w} | \mathbf{x})} \\&= \argmin_{\theta_s} \mathbb{E}_{\mathbf{w} \sim Q} \log \frac{Q_{\theta_s}(\mathbf{w} | \mathbf{x})}{P_{\theta_t}(\mathbf{w} | \mathbf{x})} \\\\
\end{align*}
$$

By logarithm's property, one might use this equivalent notation $-\log \frac{P_{\theta_t}(\mathbf{w} | \mathbf{x})}{Q_{\theta_s}(\mathbf{w} | \mathbf{x})} = \log \frac{Q_{\theta_s}(\mathbf{w} | \mathbf{x})}{P_{\theta_t}(\mathbf{w} | \mathbf{x})}$.

#### Optimization Objective

*Policy gradient*: conditioned on $\mathbf{w} \sim Q(\cdot | \mathbf{x})$ that indicates that predicted tokens are drawn from the probability distributions $Q$ given the prompts $\mathbf{x}$, included a learning rate $\eta$ such that $\eta \nabla \mathcal{J}(\theta_s)$, parameter update by gradient descent $\theta_s \leftarrow \theta_s - \eta \nabla \mathcal{J}(\theta_s)$ can get $\nabla \mathcal{J}(\theta_s)$ converged.

$r_t=\log \frac{Q_{\theta_s}(w_k | \mathbf{w}_{1:k}, \mathbf{x})}{P_{\theta_t}(w_k | \mathbf{w}_{1:k}, \mathbf{x})}$ is single-step reward,
that consider all possible tokens, there is $\sum_{w_t \in W} r_t \ge 0$.

A *regularization* term is added taking on the expectation of "mean" of $r_t$ over $T$ steps.
This is for error accumulation given first few tokens having significant impact on the succeeding token predictions,
that each prediction $p(w_k | \mathbf{w}_{1:k}, \mathbf{x})$ needs to go through its preceding tokens $\mathbf{w}_{1:k}$, and erroneous preceding tokens $\mathbf{w}_{1:k}$ result in accumulated errors for next token prediction.

This term $\mathbb{E}_{w_t \sim Q(t)} \big(r_t\big)$ refers to reward expectation at each step $t$ disregarded of preceding tokens $\mathbf{w}_{1:k}$, thereby having remediated the issue where preceding tokens $\mathbf{w}_{1:k}$ hold strong sway on the objective $\mathcal{J}(\theta_s)$.

$\sum_{k=t}^T r_t$ grows as token sequence gets long ($T$ is large).
To $\min_{\theta} \mathcal{J}(\theta_s)$, optimization tends to converge to predict short token sequence ($T$ is small).
To prevent this problem from happening, a normalization term is added $\frac{1}{T-t-1} \sum_{k=t}^T r_t$.

$$
\nabla \mathcal{J}(\theta_s) =
\underbrace{
\mathbb{E}_{\mathbf{w} \sim Q(\cdot | \mathbf{x})} \Bigg(
\sum_{t=1}^T \bigg( \frac{1}{T-t-1} \sum_{k=t}^T \underbrace{\log \frac{Q_{\theta_s}(w_k | \mathbf{w}_{1:k}, \mathbf{x})}{P_{\theta_t}(w_k | \mathbf{w}_{1:k}, \mathbf{x})}}_{:= r_t} \bigg)
\nabla Q_{\theta_s}(w_k | \mathbf{w}_{1:k}, \mathbf{x}) \Bigg)}_{\text{Policy Gradient}}+\underbrace{\mathbb{E}_{\mathbf{w} \sim Q(\cdot | \mathbf{x})} \bigg(
    \sum_{t=1}^T \nabla \mathbb{E}_{w_t \sim Q(t)} \big(r_t\big)
\bigg)}_{\text{Regularization}}
$$

### Practices

General approach is that, prepare a param-randomly-init model as a student, take an existing pre-trained model as the teacher, then train the student model with same dataset (for example, Toronto Book Corpus and English Wikipedia for BERT) to conform with teacher's outputs/token probability distribution.

Reference: https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/train.py

```python
from distiller import Distiller

teacher = teacher_model_class.from_pretrained(args.teacher_name, output_hidden_states=True)
student = student_model_class.from_pretrained(args.student_name, output_hidden_states=True)

# DISTILLER #
torch.cuda.empty_cache()
distiller = Distiller(
    params=args, 
    dataset=dataset_Train, 
    token_probs=token_probs, 
    student=student, 
    teacher=teacher
)
distiller.train()
```

Tricks:

For example, for a student `BERT-distilled` to learn from a teacher `BERT-large`, 

* Weight init is crucial to good distillation, init `BERT-distilled` with partially same params as that of `BERT-large`, e.g, copy the same layers of `BERT-large`
* Both student and teacher should see same input, copy the same embeddings from teacher and freeze the embedding layer for student
