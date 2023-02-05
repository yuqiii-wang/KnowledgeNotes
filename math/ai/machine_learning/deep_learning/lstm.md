# Long-Short Term Memory (LSTM)

## Recurrent Neural Network (RNN)

Given $x_t \in \mathbb{R}^{d}$ as the input vector at the time $t$, of $d$ dimensions/features,

define $h_t \in \mathbb{R}^{h}$ as the $t$-th time hidden layer output, and $y_t \in \mathbb{R}^{l}$ as the final semantic output,

there are

* $W_h \in \mathbb{R}^{h \times d}$ and $W_y \in \mathbb{R}^{l \times h}$ is the weight matrix for $x_t$ and $h_t$, respectively.
* $U_h \in \mathbb{R}^{h \times h}$ or $U_h \in \mathbb{R}^{l \times l}$ is the weight matrix for, either the previous time $t-1$ layer or output (depending on implementation by Elman or Jordan). 
* $b_h \in \mathbb{R}^{h}$ and $b_y \in \mathbb{R}^{l}$ are bias vectors

### Elman network

$$
\begin{align*}
h_t &= \sigma_h(W_h x_t + U_h h_{t-1} + b_h)
\\
y_t &= \sigma_y(W_y h_t + b_y)
\end{align*}
$$

### Jordan network

$$
\begin{align*}
h_t &= \sigma_h(W_h x_t + U_h y_{t-1} + b_h)
\\
y_t &= \sigma_y(W_y h_t + b_y)
\end{align*}
$$

### Training

RNN works on dataset with periodic hidden patterns, so that each sample $x_i$ should be the size of $m \times n \times \tau$, where $0 \le t \le \tau$, where $\tau$ is the pattern period, over which $U_h$ is tracing back. 

For example, a dataset over one year (365 days) is composed of tick data with hour-level granularity, and exhibits a periodic pattern of one day. Each sample should be of the size $m \times n \times 24$. 

This means, if $\tau$ is known, use it; do not partition dataset into smaller pieces.

### Discussions

RNN's secrete is $U_h$ that keeps tracking the previous output $h_{t-1}$ or $y_{t-1}$. This design has a gradient vanishing problem if $t \rightarrow \infty$, there is $\Delta U_h \rightarrow 0$ in back-propagation.

RNN's $U_h$ holds information for all previous time step, which might not reflect how real world information flows.

## Long-Short Term Memory (LSTM)

$$
\begin{align*}
f_t &= \sigma_g(W_f x_t + U_f h_{t-1} + b_f)
\\
i_t &= \sigma_g(W_i x_t + U_i h_{t-1} + b_i)
\\
o_t &= \sigma_g(W_o x_t + U_o h_{t-1} + b_o)
\\
\hat{c}_t &= \sigma_c(W_c x_t + U_c h_{t-1} + b_c)
\\
c_t &= f_t \odot c_{t-1} + i_t \odot \hat{c}_t 
\\
h_t &= o_t \odot \sigma_h(c_t)   
\end{align*}
$$

where
* $\odot$ denotes the Hadamard product (element-wise product)
* $x_t \in \mathbb{R}^d$ is an input vector
* $f_t \in (0,1)^h$ is a forget gate's activation vector
* $i_t \in (0,1)^h$ is an input/update gate's activation vector
* $o_t \in (0,1)^h$ is an output gate's activation vector 
* $h_t \in (-1,1)^h$ is a hidden state vector also known as output vector of the LSTM unit
* $\hat{c}_t \in (-1,1)^h$ is a cell input activation vector
* $c_t \in \mathbb{R}^{h}$ is a cell state vector
* $W \in \mathbb{R}^{h \times d}, U \in \mathbb{R}^{h \times h}, b_t \in \mathbb{R}^{h}$ are weight matrices and bias vector parameters

### Discussions

RNN suffers from two problems: vanishing gradient and indiscriminate retention of information over the time. LSTM alleviates with the design of a cell state $c_t$.

$\hat{c}_t \in (-1,1)^h$ is similar to the RNN's memory unit that holds information over a period of time. $f_t \odot \hat{c}_{t-1}$ removes old memory and $i_t \odot \hat{c}_t$ adds new memory. $c_t$ as the sum of the two results can summarize the long-term memory (only long term memory can survive in $f_t \odot \hat{c}_{t-1}$) and short-term memory (introduced in $i_t \odot \hat{c}_t$).

Besides, $c_t$ employs addition as the operation that helps mitigate the vanishing gradient issue caused by long chained multiplications. 

## LSTM Back Propagation

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
