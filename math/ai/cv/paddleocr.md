# Paddle OCR (Optical Character Recognition)

Paddle is Baidu developed deep learning framework.
Paddle OCR is Baidu implementation of OCR task, characterized by light weight and high accuracy.

By Dec 2024, Paddle OCR v3 is the latest implementation.

Reference:

* https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/PP-OCRv3_introduction_en.md
* https://zhuanlan.zhihu.com/p/511564666
* https://arxiv.org/pdf/2205.00159

In summary, there are two major tasks:

* Text Detection (segmentation by bounding box contained text)
* Text Recognition (visual to text)

## LK-PAN: Large Kernel Path Aggregation Network

LK-PAN (Large Kernel PAN) is a lightweight PAN (Path Aggregation Network) for instance segmentation aiming to predict class label and pixelwise instance mask to localize varying numbers of instances presented in images.

Reference:

* https://arxiv.org/pdf/2206.03001
* https://arxiv.org/pdf/1803.01534

### Inspiration/motivation: Feature Pyramid Network (FPN)

Reference:

* https://arxiv.org/pdf/1612.03144

Pyramid refers to various image sizes by up/down-sampling so that convolution kernels can capture different scale visual features.
It makes small feature proposals assigned to low feature levels and
large proposals to higher ones.

FPN has two pyramids:

* Bottom-up pathway: a feed-forward ConvNet with a scaling step of $2$, where each layer's outputs are stored for feature reference (lateral connection to latter top-down pathway)
* Top-down pathway and lateral connections: top-down pathway hallucinates higher resolution features by up-sampling spatially coarser, but semantically stronger, feature maps from higher pyramid levels.

"Hallucination" means that higher layer outputs are "guessed" by up-sampling （e.g., bi-linear interpolation） for the interpolated pixels are not contained strong semantics.
The guess/up-sampling is corrected by lateral connection.

<div style="display: flex; justify-content: center;">
      <img src="imgs/paddleocr_fpn.png" width="30%" height="35%" alt="paddleocr_fpn" />
</div>
</br>

where the lateral connection on the layer $C_l$ is 

$$
C^{\text{top-down}}_l=\text{Upsample}(C^{\text{top-down}}_{l+1}) + \text{Conv}_{1 \times 1}(C^{\text{bottom-up}}_{l})
$$

The $+$ is element-wise addition for the up-sampled $l+1$-th layer has the same image size as the $l$-th layer's.

FPN uses ResNet (basic block) as the backbone (assumed $x$ is the original image input)

A basic residual block contains

* $3 \times 3$ Convolution Layers.
* Batch Normalization (BN) and ReLU activation.
* Shortcut (residual) connections to mitigate the vanishing gradient problem.

$$
F(x, W) = \underbrace{\text{ReLU}(\text{BN}(\text{Cov}_{3 \times 3}}_{\text{the }l_{+1}\text{-th layer output}}(
    \underbrace{\text{ReLU}(\text{BN}(\text{Cov}_{3 \times 3}(x)))}_{\text{the }l\text{-th layer output}}))) + x
$$

|Feature Level|Stage|Resolution ($H \times W$)|Output Channels|Sampling|
|-|-|-|-|-|
|$C_1$|Initial Conv + Pooling|$\frac{1}{4} \times x$|64|2x down-sampling|
|$C_2$|conv2_x|$\frac{1}{4} \times x$|256|no down-sampling|
|$C_3$|conv3_x|$\frac{1}{8} \times x$|512|2x down-sampling|
|$C_4$|conv4_x|$\frac{1}{16} \times x$|1024|2x down-sampling|
|$C_5$|conv5_x|$\frac{1}{32} \times x$|2048|2x down-sampling|

### The LK-PAN has below enhancements based on FPN

(a) FPN backbone
(b) Bottom-up path augmentation
(c) Adaptive feature pooling
(d) and (e) Box branch and Fully-connected fusion

<div style="display: flex; justify-content: center;">
      <img src="imgs/paddleocr_kl_pan.png" width="70%" height="30%" alt="paddleocr_kl_pan" />
</div>
</br>

#### Bottom-up Path Augmentation

Besides the FPN existing lateral connection, LK-PAN added additional lateral connections in the bottom-up process.

This augmentation works for the layers $\{C_2, C_3, C_4, C_5\}$, or denoted as $\{P_2, P_3, P_4, P_5\}$ in KL-PAN.
They are the same such that $C_l=P_l$.

Let $\{N_2, N_3, N_4, N_5\}$ be the newly generated feature maps in KL-PAN corresponding to $\{P_2, P_3, P_4, P_5\}$.

The $l+1$-th layer feature maps in KL-PAN are generated by

$$
N_{l+1}=\text{Conv}_{3 \times 3}\big(\text{Conv}_{3 \times 3}(N_l, \text{stride}=2) + P_{l+1}\big)
$$

#### Adaptive Feature Pooling

It is a simple component to aggregate features from all feature levels for each proposal.

ROIAlign is used to pool feature grids from each level.
Then a fusion operation (element-wise max or sum) is utilized to fuse feature grids from different levels.

Up-sampling (e.g., by bi-linear interpolation) is used to align image scales.

#### Box Branch and Fully-Connected Fusion

LK-PAN referenced Mask R-CNN and 

## DBNet (Differentiable Binarization Net)

DBNet performs binarization process in a segmentation network.
Optimized along with a Differentiable Binarization (DB) module, a segmentation network can adaptively set the thresholds for binarization, which not only simplifies the post-processing but also enhances the performance of text detection.

Reference:

* https://github.com/MhLiao/DB
* https://arxiv.org/pdf/1911.08947
* https://arxiv.org/pdf/2202.10304

Illustrated in below, DBNet is basically an enhancement (red flow) to segmentation so that the areas of interest can be more likely got picked up.

<div style="display: flex; justify-content: center;">
      <img src="imgs/paddleocr_dbnet.png" width="70%" height="30%" alt="paddleocr_dbnet" />
</div>
</br>

Given a pixel $p_{i,j}$ on an image $\mathbb{R}^{H \times W}$, standard binarization is simply a step function returning a value either $0$ or $1$ conditioned on a threshold $t$:

$$
p_{i,j} = \begin{cases}
  1 & \text{if } p_{i,j} < t, \\
  0 & \text{if } \text{otherwise}
\end{cases}
$$

This step function is not differentiable hence not trainable.
The static threshold $t$ is applied to global image $\mathbb{R}^{H \times W}$ rather than discriminative between local image areas.

DBNet proposes a novel binarization function

$$
\hat{p}_{i,j} =
\frac{1}{1+e^{-k(p_{i,j}-t_{i,j})}}
$$

where $t_{i,j}$ is the adaptive threshold map learned from the network; $k$ indicates the amplifying factor. $k$ is set to $50$ empirically.

The final result is as below Standard Binirization (SB) vs Differentiable Binirization (DB).

<div style="display: flex; justify-content: center;">
      <img src="imgs/paddleocr_db_step_func.png" width="20%" height="30%" alt="paddleocr_db_step_func" />
</div>
</br>

## SVTR

<div style="display: flex; justify-content: center;">
      <img src="imgs/paddleocr_svtr.png" width="70%" height="30%" alt="paddleocr_svtr" />
</div>
</br>