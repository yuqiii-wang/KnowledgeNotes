# Sequence NNs: RNN, GRU and LSTM

## Recurrent Neural Network (RNN)

Given $\bold{x}_t \in \mathbb{R}^{d}$ as the input vector at the time $t$, of $d$ dimensions/features,

define $\bold{h}_t \in \mathbb{R}^{h}$ as the $t$-th time hidden layer output, and $\bold{y}_t \in \mathbb{R}^{l}$ as the final semantic output,

there are

* $W_h \in \mathbb{R}^{h \times d}$ and $W_y \in \mathbb{R}^{l \times h}$ is the weight matrix for $\bold{x}_t$ and $\bold{h}_t$, respectively.
* $U_h \in \mathbb{R}^{h \times h}$ or $U_h \in \mathbb{R}^{l \times l}$ is the weight matrix for, either the previous time $t-1$ layer or output (depending on implementation by Elman or Jordan). 
* $\bold{b}_h \in \mathbb{R}^{h}$ and $\bold{b}_y \in \mathbb{R}^{l}$ are bias vectors

Given a sequence length of $T$ such that $t \in \{ 1, 2, ..., T \}$, the temporary memory of an RNN consists of

* $\bold{h} \in \mathbb{R}^{T \times h \times d}$ the hidden neurons. It stores a length of $T$'s memory $\bold{h}=\{ \bold{h}_1, \bold{h}_2, ..., \bold{h}_t, ..., \bold{h}_{T} \}$.

### Elman network

$$
\begin{align*}
\bold{h}_t &= \sigma_h(W_h \bold{x}_t + U_h \bold{h}_{t-1} + \bold{b}_h)
\\
\bold{y}_t &= \sigma_y(W_y \bold{h}_t + \bold{b}_y)
\end{align*}
$$

### Jordan network

$$
\begin{align*}
\bold{h}_t &= \sigma_h(W_h \bold{x}_t + U_h \bold{y}_{t-1} + \bold{b}_h)
\\
\bold{y}_t &= \sigma_y(W_y \bold{h}_t + \bold{b}_y)
\end{align*}
$$

### Semantics

RNN works on dataset with periodic hidden patterns, so that each sequence sample $X = \{ \bold{x}_1, \bold{x}_2, ..., \bold{x}_t, ..., \bold{x}_{T} \}$ should be the size of $T \times h \times d$.

In practice, $T$ should represent a complete cycle of some semantics, such as $T = 1,000$ for NLP to study an article and $T = 100,000$ to study a book, or $T=24 \times 60$ for one day operation minute-level tick data.

### Forward Pass

Here sets $\sigma_h=\tanh$ and $\sigma_y=\text{softmax}$. 
The final error output is defined as cross entropy.
A forward pass (Elman network) is defined as below.

$$
\begin{align*}
\text{for each step } \bold{x}_t \text{ in } X &:
\\  && \bold{h}_t &= \tanh(W_h \bold{x}_t + U_h \bold{h}_{t-1} + \bold{b}_h)
\\  && \bold{z}_t &= W_z \bold{h}_t + \bold{b}_z
\\  && \hat{\bold{y}}_t &= \text{softmax}(\bold{z}_t) 
\\  && \mathcal{L}_t &= - \bold{y}_t^{\top} \log \hat{\bold{y}}_t
\\  && \mathcal{L} &+= \mathcal{L}_t
\end{align*}
$$

where $\mathcal{L} = \sum^{T}_{t} \mathcal{L}_t$ is the total loss over the whole sequence $X$.

Here defines $\bold{y}_t \in \mathbb{R}^n$, where for instance in BERT tokenization there is $n=30522$.
It is one-hot encoding of $t$-th sequence step chosen class (e.g., the token at the $t$-th position in a sentence).

$$
\bold{y}_{t, i=2023} = \text{OneHotEncoding}(2023) = 
[\underbrace{0, 0, 0, 0, 0, 0, 0, 0, ..., 0}_{\times 2022}, 1, \underbrace{0, 0, 0, 0, 0, 0, 0, 0, ..., 0}_{\times (30522 - 1 - 2022)}]
$$

The $\log \hat{\bold{y}}_t$ is the log probability of predicted class $\hat{\bold{y}}_t \in \mathbb{R}^n$,
and $\mathcal{L}_t = - \bold{y}_t^{\top} \log \hat{\bold{y}}_t$ gives a loss value of predicting a particular class,
for in $\bold{y}_t$, only the correct prediction yields one-hot value $1$, others are $0$s.

For example, if the $t$-th step's correct class (e.g., token in NLP) is $i=2023$, there is

$$
\begin{align*}
    \mathcal{L}_t &=
    -\bold{y}^{\top}_{t, i=2023} \log \hat{\bold{y}}_t 
\\ &= 
    (-1) \cdot (0 \cdot \log \hat{{y}}_{t, i=1} + 0 \cdot \log \hat{{y}}_{t, i=2} + ... + 1 \cdot \log \hat{{y}}_{t, i=2023} + ... + 0 \cdot \log \hat{{y}}_{t, i=30522})
\\ &= 
    -\log \hat{{y}}_{t, i=2023}
\end{align*}
$$

### Back-Propagation (Exampled on $W_z$ and $\bold{b}_z$ Updates)

$\mathcal{L}_t \in \mathbb{R}^1$ is a scalar value, and $\bold{z}_t \in \mathbb{R}^n$ is the linear/dense class prediction output awaiting normalized by softmax.
The first updates should be on $W_z$ and $\bold{b}_z$.

With the applied learning rate $\eta$, they are updated by summing up all back-propagated errors over all steps $t \in \{ 1, 2, ..., T \}$.

$$
\begin{align*}
W_z & \leftarrow 
\sum_t^{T} \eta \frac{\partial \mathcal{L}_t}{\partial \bold{h}_z} + W_z + \xi_{W_z}
&& &
\bold{b}_z & \leftarrow 
\sum_t^{T} \eta \frac{\partial \mathcal{L}_t}{\partial \bold{h}_z} + \bold{b}_z + \xi_{\bold{b}_z}
\end{align*}
$$

where $\xi_{W_z}$ and $\xi_{\bold{b}_z}$ are random noises required for SGD (Stochastic Gradient Descent).

By chain rule, there should be

$$
\frac{\partial \mathcal{L}_t}{\partial \bold{h}_z} = 
\frac{\partial \mathcal{L}_t}{\partial \hat{\bold{y}}_t}
\frac{\partial \hat{\bold{y}}_t}{\partial \bold{z}_t}
\frac{\partial \bold{z}_t}{\partial \bold{h}_z}
$$

$\frac{\partial \mathcal{L}_t}{\partial \hat{\bold{y}}_t} \in \mathbb{R}^{1 \times n}$ is the derivative with respect to each class prediction probability, where $n$ refers to the num of prediction classes, e.g., num of vocab such as $n=30522$ for BERT tokens.
For each entry $i = 1,2,...,n$, there is

$$
\frac{\partial \mathcal{L}_t}{\partial \hat{{y}}_{t,i}} =
-\sum_i^{n} y_{t,i} \frac{\partial \log \hat{{y}}_{t,i}}{\partial z_{t,j}} =
-\frac{\partial \log \hat{{y}}_{t,i}}{\partial z_{t,j}} 
\text{ for } \bold{y}_{t} \text{ is one-hot encoded}
$$

For example, $\frac{\partial \mathcal{L}_t}{\partial \hat{\bold{y}}_t}\Big|_{i=2023}$ assumes the $t$-step's true label is of $i=2023$, there is

$$
\frac{\partial \mathcal{L}_t}{\partial \hat{\bold{y}}_t} = 
[\underbrace{0, 0, 0, 0, 0, 0, 0, 0, ..., 0}_{\times 2022}, 
\frac{\partial \log \hat{{y}}_{t,i=2023}}{\partial z_{t,j}},
\underbrace{0, 0, 0, 0, 0, 0, 0, 0, ..., 0}_{\times (30522 - 1 - 2022)}]
$$

$\frac{\partial \log \hat{\bold{y}}_t}{\partial \bold{z}_t} \in \mathbb{R}^{n \times n}$ is a Jacobian. 
For each entry there is 

$$
\begin{align*}
    \frac{\partial }{\partial z_{t,j}} \Big( -\log \big( \underbrace{\text{softmax}({z}_{t,i})}_{\hat{y}_{t,i}} \big) \Big) &=
    -\frac{\partial }{\partial z_{t,j}} \Big( \log \frac{e^{z_{t,i}}}{\sum_i^n e^{z_{t,i}}} \Big) 
\\ &=
    -\frac{\partial }{\partial z_{t,j}} \Big( \log{e^{z_{t,i}}} - \log{\sum_i^n e^{z_{t,i}}} \Big)
\\ &=
    -\frac{\partial }{\partial z_{t,j}} \Big( {z_{t,i}} - \log{\sum_i^n e^{z_{t,i}}} \Big)
\\ &=
    \frac{\partial }{\partial z_{t,j}} \log{\sum_i^n e^{z_{t,i}}} - \frac{\partial z_{t,i}}{\partial z_{t,j}}
\\ &=
    \frac{1}{\sum_i^n e^{z_{t,i}}} \frac{\partial \sum_i^n e^{z_{t,i}}}{\partial z_{t,j}} - \frac{\partial z_{t,i}}{\partial z_{t,j}}
\\ &=
    \frac{1}{\sum_i^n e^{z_{t,i}}} \frac{\partial (e^{z_{t,1}}+e^{z_{t,2}}+...+e^{z_{t,n}})}{\partial z_{t,j}} - \frac{\partial z_{t,i}}{\partial z_{t,j}}
&& \text{where } 
    \frac{\partial z_{t,i}}{\partial z_{t,j}} = 
    \left\{ \begin{array}{r}
        1 \qquad i = j \\
        0 \qquad i \ne j
    \end{array}\right.
\\ &=
    \frac{e^{z_{t,i}}}{\sum_i^n e^{z_{t,i}}} - 1
&& \text{where } i = j
\\ &=
    \hat{y}_{t,i} - 1
\end{align*}
$$

So that, given assumption of one-hot encoding for $\bold{y}_t=[0,0,0,...,0,1,0,...0,0,0]$ where only true token index is marked $1$, there is

$$
\begin{align*}
\frac{\partial \mathcal{L}_t}{\partial \bold{h}_z} &= 
\frac{\partial \mathcal{L}_t}{\partial \hat{\bold{y}}_t}
\frac{\partial \hat{\bold{y}}_t}{\partial \bold{z}_t}
\frac{\partial \bold{z}_t}{\partial \bold{h}_z}
\\&=
(\hat{\bold{y}}_{t} - \bold{y}_t)
\frac{\partial \bold{z}_t}{\partial \bold{h}_z}
\\&=
(\hat{\bold{y}}_{t} - \bold{y}_t)
W_z
\end{align*}
$$

### Discussions on RNN

RNN's secrete is $U_h$ that keeps tracking the previous output $\bold{h}_{t-1}$ or $y_{t-1}$. This design has a gradient vanishing problem if $t \rightarrow \infty$, there is $\Delta U_h \rightarrow 0$ in back-propagation.

RNN's $U_h$ holds information for all previous time step, which might not reflect how real world information flows.

## Gated Recurrent Unit (GRU)

GRU is a variant of RNN but much simpler than Long Short-Term Memory (LSTM).
GRUs have two gates that control the flow of information:

Let $[...]$ be concatenation operation, $\bold{h}_{t-1}$ be previous time-step hidden state, and $\bold{x}_t$ is the current input

* Update gate ($\bold{z_t}$):

$$
\bold{z_t}=\sigma(W_z\cdot[\bold{h}_{t-1}; \bold{x}_t])
$$

* Reset gate ($\bold{r_t}$):

$$
\bold{r_t}=\sigma(W_r\cdot[\bold{h}_{t-1}; \bold{x}_t])
$$

* Candidate Hidden State ($\hat{\bold{h}}_t$)

$$
\hat{\bold{h}}_t=\sigma(W_h\cdot[\bold{r_t}; \bold{h}_{t-1}; \bold{x}_t])
$$

* Final Hidden State ($\bold{h}_t$):

$$
\bold{h}_t=(\bold{1}-\bold{z}_t)\cdot\bold{h}_{t-1}+\bold{z}_t\cdot\hat{\bold{h}}_t
$$

### Discussions on GRU

Update gate ($\bold{z_t}$) is used to control the ratio between previous hidden state $\bold{h}_{t-1}$ vs this time hidden state $\bold{h}_{t}$.

## Long-Short Term Memory (LSTM)

$$
\begin{align*}
f_t &= \sigma_g(W_f \bold{x}_t + U_f \bold{h}_{t-1} + b_f)
\\
i_t &= \sigma_g(W_i \bold{x}_t + U_i \bold{h}_{t-1} + b_i)
\\
o_t &= \sigma_g(W_o \bold{x}_t + U_o \bold{h}_{t-1} + b_o)
\\
\hat{c}_t &= \sigma_c(W_c \bold{x}_t + U_c \bold{h}_{t-1} + b_c)
\\
c_t &= f_t \odot c_{t-1} + i_t \odot \hat{c}_t 
\\
\bold{h}_t &= o_t \odot \sigma_h(c_t)   
\end{align*}
$$

where

* $\odot$ denotes the Hadamard product (element-wise product)
* $\bold{x}_t \in \mathbb{R}^d$ is an input vector
* $f_t \in (0,1)^h$ is a forget gate's activation vector
* $i_t \in (0,1)^h$ is an input/update gate's activation vector
* $o_t \in (0,1)^h$ is an output gate's activation vector 
* $\bold{h}_t \in (-1,1)^h$ is a hidden state vector also known as output vector of the LSTM unit
* $\hat{c}_t \in (-1,1)^h$ is a cell input activation vector
* $c_t \in \mathbb{R}^{h}$ is a cell state vector
* $W \in \mathbb{R}^{h \times d}, U \in \mathbb{R}^{h \times h}, b_t \in \mathbb{R}^{h}$ are weight matrices and bias vector parameters

### Discussions on LSTM

RNN suffers from two problems: vanishing gradient and indiscriminate retention of information over the time. LSTM alleviates with the design of a cell state $c_t$.

$\hat{c}_t \in (-1,1)^h$ is similar to the RNN's memory unit that holds information over a period of time. $f_t \odot \hat{c}_{t-1}$ removes old memory and $i_t \odot \hat{c}_t$ adds new memory. $c_t$ as the sum of the two results can summarize the long-term memory (only long term memory can survive in $f_t \odot \hat{c}_{t-1}$) and short-term memory (introduced in $i_t \odot \hat{c}_t$).

Besides, $c_t$ employs addition as the operation that helps mitigate the vanishing gradient issue caused by long chained multiplications. 

### LSTM Back Propagation

reference: https://www.geeksforgeeks.org/lstm-derivation-of-back-propagation-through-time/

Define an error (MSE) from the layer's output, and the $t$-th layer's gradient.
$$
\begin{align*}
    {e} &= \big( {y} - {h}({x}) \big)^2 \\
    \delta{e}  &= \frac{\partial{e}}{\partial {h}_t}
\end{align*}
$$

Gradient with respect to output gate  
$$
\frac{\partial e}{\partial o_t} = 
\frac{\partial{e}}{\partial {h}_t} \cdot
\frac{\partial {h}_t}{\partial {o}_t} =
\delta{e} \cdot  \tanh(c_t)
$$
