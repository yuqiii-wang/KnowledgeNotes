# Attention Is All You Need

## A Quick Attention Overview

* Attention

Given $Q$ for query, $K$ for key, $V$ for value, a simple self-attention can be computed as

$$
\text{Attention}(Q,K,V) = \text{softmax} \Big( \frac{Q K^{\top}}{\sqrt{d_k}} \Big) V
$$
where $\text{softmax} (\bold{x}) = \frac{e^{\bold{x}}}{\sum^K_{k=1}e^{\bold{x}}}$ in which $\bold{x}=\frac{Q K^{\top}}{\sqrt{d_k}}$.

$d_k$ is the dimension of query $Q$ and key $K$.
Define the dimension of value $V$ as $d_v$ (value is often regarded as outputs).

* Multi-Head Attention

For multi-head attention, there is

$$
\text{MultiHeadAttention}(Q,K,V) = \text{concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h) W
$$
where $\text{head}_i = \text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$.

The weights are $W \in \mathbb{R}^{h \cdot d_v \times d_{model}}, W_i^Q \in \mathbb{R}^{d_{model} \times d_k}, W_i^K \in \mathbb{R}^{d_{model} \times d_k}, W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, where $d_{model}$ is the dimension of one single-attention head.

For example, in BERT base, there are $h=12$ attention heads ( $h = d_{model} / d_k = 768 / 64 = 12$); in BERT Large, there are $h=16$ attention heads ( $h = d_{model} / d_k = 1024 / 64 = 16$ ).
The choice of $d_{model} = 768$ is the result of employing wordpiece embedding per vocab.

* Feed-Forward Network (FFN)

Define a Feed-Forward Network (FFN), which is a $4$-times dimension increased fully connected network, such as in BERT base, there is `feed_forward_dim=3072` by $3072 = 4 \times 768$.
The activation function is a simple ReLU $\sigma(x) = max(0, x)$.

$$
FFN(\bold{x}) = \max(0, \bold{x}W_1 + \bold{b}_1)W_2 + \bold{b}_2
$$

where one token $\bold{x} \in \mathbb{R}^{1 \times d_{model}}$ is passed to $FFN(\bold{x})$, in which $W_1 \in \mathbb{R}^{4 d_{model} \times d_{model}}$ and $W_2 \in \mathbb{R}^{ d_{model} \times 4d_{model}}$. 

## Inspiration

Attention solves **long-distance dependency** issue haunting LSTM and RNN.

The predecessor LSTM (Long Short Term Memory) and GRU (Gated Recurrent Unit) are capable of learning latent info about sequence but have some disadvantages.
* have many step functions/gates that are not differentiable; this causes swings between on/off states that drop info; attention replaces with differentiable softmax
* long sequence info drops for vanishing gradients over the sequence; attention implements residual addition to amplify errors.
* LSTM has much more parameters to train, hence hard to train and difficult to explain/interpret the model