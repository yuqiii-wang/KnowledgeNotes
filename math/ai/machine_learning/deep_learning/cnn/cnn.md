# Convolutional Neural Network (CNN)

## Convolutional Layer

### Convolution Forward and Back Propagation

Given an input image $X$ and a filter $F$, one forward pass of convolution is $O = X \otimes F$

$$
\begin{bmatrix}
    O_{11} & O_{12} \\
    O_{21} & O_{22}
\end{bmatrix} =
\begin{bmatrix}
    X_{11} & X_{12} & X_{13} \\
    X_{21} & X_{22} & X_{23} \\
    X_{31} & X_{32} & X_{33}
\end{bmatrix} \otimes
\begin{bmatrix}
    F_{11} & F_{12} \\
    F_{21} & F_{22}
\end{bmatrix}
$$

unfold the $\otimes$ operator, there are

$$
\begin{align*}
O_{11} &= X_{11} F_{11} + X_{12} F_{12} + X_{21} F_{21} + X_{22} F_{22} \\
O_{12} &= X_{12} F_{11} + X_{13} F_{12} + X_{22} F_{21} + X_{23} F_{22} \\
O_{21} &= X_{21} F_{11} + X_{22} F_{12} + X_{31} F_{21} + X_{32} F_{22} \\
O_{22} &= X_{22} F_{11} + X_{23} F_{12} + X_{32} F_{21} + X_{33} F_{22} \\
\end{align*}
$$

The back propagation of $F_{11}$ given loss $\mathcal{L}$ is

$$
\frac{\partial \mathcal{L}}{\partial F_{11}} =
\frac{\partial \mathcal{L}}{\partial O_{11}} X_{11} +
\frac{\partial \mathcal{L}}{\partial O_{12}} X_{12} +
\frac{\partial \mathcal{L}}{\partial O_{21}} X_{21} +
\frac{\partial \mathcal{L}}{\partial O_{22}} X_{22}
$$

### Other Setups in A Convolutional Layer

#### Stride

Skip a number of $s$ pixels then do next convolution.

It can break spatial info as feature points from adjacent convolutions are likely together contribute the same semantic visual feature.

Use large stride when image resolution is high; small when low.

#### Padding

#### Pooling

### Typical Computation Cost of A Convolutional Layer

* Filter kernel size: $k \times k$
* Image size $m \times n$ divided by stride $s$: $\frac{m \times n}{s \times s}$
* The number of filters/channels $c$

Total: $(k \times k) \times \frac{m \times n}{s \times s} \times c$

### Calculation Example

Given an image $224 \times 224 \times 3$, consider $11 \times 11$ kernel with stride by $4$,

* Num convolutions over a row/col: $56=224/4$

## Convolutional Neural Network (CNN) vs Fully Convolutional Network (FCN)

CNN includes fully connected layers that result in loss of image spatial info but that is preserved in FCN.

||CNN|FCN|
|-|-|-|
|Arch|Combines convolutional and pooling layers, **followed by fully connected layers**|Consists only of convolutional, pooling, and upsampling layers.|
|Tasks|Image classification, object detection (with post-processing like bounding boxes).|Semantic segmentation, dense prediction, depth estimation, super-resolution tasks.|

## AlexNet

<div style="display: flex; justify-content: center;">
      <img src="imgs/alexnet.png" width="30%" height="60%" alt="alexnet" />
</div>
</br>

### Why convolution then pooling, and why three convolutions then pooling

Intuitively,

* First and second convolution then pooling: contained large kernels $11 \times 11$ and $5 \times 5$, useful to quickly locate local image features and reduce image resolution; large kernels are good to capture semantic features.
* Multi-convolution then pooling: used small kernel of $3 \times 3$ for 3 times consecutively that enables learning highly abstract features; pooling discards spatial details, so delaying pooling allows the network to retain more fine-grained information for complex abstractions.

### Why two dense layers before the output dense layer

A dense layer is a fully connected layer $\bold{y}=\sigma(W\bold{x}+\bold{b})$.

Intuitively,

* First Dense Layer: Reshapes/flattens low-level features $256 \times 6 \times 6$ to a vector of size $9216$ higher-level abstractions. Activations primarily capture feature combinations (e.g., patterns like "edges forming an object").
* Second Dense Layer: Refines these abstractions into class-specific patterns, applying non-linear transformations to aggregate and focus the receptive fields.
* Third Dense Layer: output transform

Mathematical Insight:

* First Dense Layer: Local receptive field combinations (flattened convolutional outputs by $W_1 \in \mathbb{R}^{4096 \times 9216}$).
* Second Dense Layer: Global object representations (class scores by $W_2 \in \mathbb{R}^{4096 \times 4096}$).
* Third Dense Layer: output transform by $W_3 \in \mathbb{R}^{1000 \times 4096}$ with activation by softmax

Empirical study:

More dense layers can see $\text{ReLU}$ saturation that many activation values tend to get very large or zero, signaled redundancy in containing too many neurons (two layers are enough).

### Indicators of Saturation

#### Empirical Observations

* Training and validation accuracy/loss plateau even with extended training
* Larger/more convolution kernels do not yield better results
* Getting deeper/more layers does not give better results
* Getting wider/larger weight matrix/more neuron per layer does not give better results
* Having small/even no stride does not give better results

#### Theoretical Indicators

* If weight matrices $W$ have **small eigenvalues**, the weight matrix may not be effectively transforming the input space.
* If weight matrices $W$ have **large eigenvalues**, the transformations may be overly redundant or lead to gradient instability.
* A large fraction of neurons consistently output zero (dead neurons in ReLU layers), indicating wasted capacity.

Recall linear algebra that $W\bold{x}=\lambda\bold{x}$ means transforming input $\bold{x}$ by $W$ is same as getting scaled by $\lambda$.

If $\lambda \gg 0$, it leads to excessive amplification of inputs $\bold{x}$ along certain directions.

* Gradient Instability: Large eigenvalues propagate large gradients during back-propagation, which can destabilize training.
* Redundancy: Over-amplifying features may result in over-fitting or redundant transformations.

If $\lambda \approx 0$, it means low rank, not utilizing all degrees of freedom in the input.

* Some features are being ignored or not contributing to the output.

#### Human Evaluation

* Receptive Field Analysis: for low-level features are semantic to human understanding, one can manually review the convolution results of the first layer; if objects have too many filters repeatedly focusing on the same areas extracting similar features, it signals saturation.
