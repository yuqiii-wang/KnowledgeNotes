# Bundle Adjustment Optimization

To facilitate computation, only partial camera poses and landmark observations are selected for Bundle Adjustment (BA), rather than including the whole history poses and existing landmarks.

## Keyframe

Each recorded camera pose is called a *frame*. 
*Keyframe* refers to selected frames with rich and distinct information that best help BA to fast compute the poses almost the same as the results by all frames.

### Keyframe Selection Methods

Denote frame-indexed feature vectors (intuitively speaking, a full video film) $\mathbf{V}=\{v_i : i=1,2,...,N\}$; a segment of the video can be defined $\mathbf{Q}=\{v_i: i=l,...,r\}\subset\mathbf{V}$. The remaining segments (take away $\mathbf{Q}$ from $\mathbf{V}$) can be defined as $\overline{\mathbf{Q}}=\mathbf{V} \setminus \mathbf{Q}$, where $\setminus$ denotes *set minus* operator.

* Similarity Based 

The similarity between any two frames can be calculated using a distance measure $d(v_i, v_j)$ that gives a high value output if the two frames are similar. Here define two types of similarities $S$ and $C$.

Self-similarity (compare $v_k$ with frames' feature vector in a segment $\mathbf{Q}$):

$$
S(v_k, \mathbf{Q}) = 
\frac{1}{|\mathbf{Q}|} \sum_{v_k, v_i \in \mathbf{Q}} d(v_k, v_i)
$$

Cross-similarity (compare $v_k$ with frames' feature vector from other segments $\overline{\mathbf{Q}}$):

$$
C(v_k, \overline{\mathbf{Q}}) = 
\frac{1}{| \overline{\mathbf{Q}}|} \sum_{v_k \in \mathbf{Q}, v_i \in \overline{\mathbf{Q}}} d(v_k, v_i)
$$

A good keyframe should have a high $S$ score ($v_k$ being very similar to the other frames' feature vectors within the same segment) and low $C$ score ($v_k$ being distinctive among frames in other segments).

* Linear Discriminant Analysis (LDA)

Define a segmentation method to $\mathbf{V}$, such that $\mathbf{V}$ is partitioned into $K$ non-overlapping sets of contiguous frames. Each $\mathbf{Q}_j$ has $N_k$ frames.

$$
\mathbf{V} = 
\bigcup_{j=1,2,...,m} \mathbf{Q}_k
$$

First compute mean feature vector $\mu_j$ from each $\mathbf{Q}_j$ and the global mean $\mu$ from $\mathbf{V}$, then compute the within-class $S_w$ and between-class $S_b$ variances.

$$
\begin{align*}
    S_w &= \sum^m_{j=1} \sum_{v_i \in \mathbf{Q}_j}
    (v_i - \mu_j)(v_i - \mu_j)^\text{T}
    \\\\
    S_b &= \sum^m_{j=1} 
    N_j (\mu_j-\mu)(\mu_j-\mu)^\text{T}
\end{align*}
$$

The optimization becomes
$$
\mathbf{W}^* = arg \space \underset{\mathbf{W}}{max} 
\frac{\mathbf{W}^\text{T}S_b\mathbf{W}}{\mathbf{W}^\text{T}S_w\mathbf{W}}
$$

Now project the source feature space $\mathbf{V}$ by $\mathbf{W}^*$ to a lower dimensional space $\~\mathbf{V}$, there is $\~\mathbf{V}=\mathbf{W}^{*\text{T}}\mathbf{V}$. The projected space $\~\mathbf{V}$ should see each $\mathbf{Q}_j$'s centroid/mean $\mu_j$ widely separated and frames' features $v_i$ within each $\mathbf{Q}_j$ concentrated.

The best keyframe feature vector $v_k^*$ in each segment $\mathbf{Q}_j$ should be the one frame's feature vector closest to the $\mathbf{Q}_j$'s centroid/mean $\mu_j$ in the projected space $\~\mathbf{V}$.

$$
v^*_k = arg \space \underset{v_k \in \mathbf{Q}_k}{min} \space
\big|\big|
    \mathbf{W}^\text{T} (v_k - \mu_j)
\big|\big|
$$

## Co-Visibility

The so-called *co-visibility* refers to those features that are observed together with the current keyframe.

### Co-Visibility in Schur Elimination

Recall that in Schur elimination, the camera pose computation gives the below expression. 

$$
(\mathbf{B}-\mathbf{E}\mathbf{C}^{-1}\mathbf{E}^\text{T})
\Delta \mathbf{x}_{\mathbf{\xi}}=
\mathbf{v} - \mathbf{E}\mathbf{C}^{-1} \mathbf{w}
$$

Here denote $\mathbf{S}=\mathbf{B}-\mathbf{E}\mathbf{C}^{-1}\mathbf{E}^\text{T}$.

The non-zero matrix block on the off-diagonal line of the $\mathbf{S}$ matrix indicates that there is a co-observation between the two camera variables. It is called co-visibility.

If $\mathbf{S}=\mathbf{0}$, there is no shared features observed at these different camera poses.

If $\mathbf{S} \ne \mathbf{0}$ such as the figure given below, $\mathbf{S}$ illustrates the co-visibility of features. The zoomed-in sub-square matrix $C_1-C_4$ indicates that, the camera poses $C_1$ and $C_2$ see the same features as that from camera pose $C_4$, and vice versa. Camera pose $C_3$ sees no shared features.

![schur_coeff](imgs/schur_coeff.png "schur_coeff")

## Sliding Window Method

### Motivations

In a real-world scenario, often a camera can only see a number of landmarks at a pose/keyframe, and the nearby poses/keyframes are likely able to see almost the same number of features/landmarks.

A sliding window aims to group these neighboring poses/keyframes and the associated landmarks. BA only performs computation on the selected poses and landmarks in this window.

### Definition

Now consider a sliding window. Assume there are $n$ keyframes in this window, and
their poses are denoted as $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n$.

Given the same sliding window containing the aforementioned poses, suppose there are $m$ landmarks in this window $\mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_m$.

The conditional distribution of the poses $\mathbf{x}_k$ conditioned on $\mathbf{y}_k$ can be expressed as below under Gaussian noise assumption.

$$
[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n | \mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_m]
\sim N([\mathbf{\mu}_1, \mathbf{\mu}_2, ..., \mathbf{\mu}_n]^\text{T}, \Sigma_{n})
$$

where $\mathbf{\mu}_k$ is mean of the $k$-th keyframe, $\Sigma$ is the covariance matrix of all keyframes.

BA collectively computes the windows's keyframes, and delivers the $\mathbf{S}$ that determines camera poses $\mathbf{x}_{\mathbf{\xi}_k}: k = 1,2,...,n$.

### Manage Keyframes in a Window

* Adding New Keyframes

The sliding window has established $n$ keyframes at the last moment, and a certain Gaussian distribution describes poses conditional on landmarks.

A new keyframe can be directly added to the window (denoted as $\mathbf{x}_{n+1}$) and BA can normally perform computation on $\mathbf{S}$, in contrast to pose removal that has concerns over correlated observed landmarks between various poses.

$$
[\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n, \mathbf{x}_{n+1} | \mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_m, \mathbf{y}_{m+1}, ..., \mathbf{y}_{m_{n+1}}]
\sim N([\mathbf{\mu}_1, \mathbf{\mu}_2, ..., \mathbf{\mu}_n, \mathbf{\mu}_{n+1}]^\text{T}, \Sigma_{n+1})
$$

where $[\mathbf{y}_{m+1}, ..., \mathbf{y}_{m_{n+1}}]$ are the new landmarks observed from the new pose $\mathbf{x}_{n+1}$. There is possibility that landmarks $\mathbf{y}_k \in [\mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_m]$ are observed from the new camera pose $\mathbf{x}_{n+1}$

* Removing Old Keyframes

Keyframe removal, intuitively speaking, asks for the result of marginalization of $\mathbf{x}_1$ such as $P(\mathbf{x}_2, \mathbf{x}_3, ..., \mathbf{x}_n, \mathbf{y}_{1}, \mathbf{y}_{2}, ..., \mathbf{y}_{m} | \mathbf{x}_1)$, where $[\mathbf{y}_{1}, \mathbf{y}_{2}, ..., \mathbf{y}_{m}]$ are shared landmark observations between camera poses $[\mathbf{x}_2, ..., \mathbf{x}_n]$.

Schur elimination (a.k.a marginalization) removing $\mathbf{x}_1$ can cause a sparse matrix dense. This phenomenon is termed *fill-in*.

Below is an example, where a matrix $\mathbf{\Lambda}$ is composed of $\mathbf{B}=\Lambda_m \in \mathbb{R}^{m \times m}, \mathbf{E}=\Lambda_{mp} \in \mathbb{R}^{m \times p}, \mathbf{C}=\Lambda_p \in \mathbb{R}^{p \times p}$. 

$$
\begin{bmatrix}
    \mathbf{B} & \mathbf{E} \\\\
    \mathbf{E}^\text{T} & \mathbf{C}
\end{bmatrix}
\begin{bmatrix}
    \Delta \mathbf{x}_{\mathbf{\xi}} \\\\
    \Delta \mathbf{x}_{\mathbf{p}}
\end{bmatrix}=
\begin{bmatrix}
    \mathbf{v} \\\\
    \mathbf{w}
\end{bmatrix}
$$

![marginalization_s](imgs/marginalization_s.png "marginalization_s")

First, permutation takes place moving $\mathbf{x}_1$- related landmark elements to the margin from $\mathbf{E}, \mathbf{E}^\text{T}, \mathbf{C}$.

Then perform marginalization. Denote the permuted sub matrices (marked as slash-shaded areas) as $\mathbf{B}'=\mathbf{\Lambda}_{p_{11}}, \mathbf{E}'=[\mathbf{\Lambda_{mp_{1,1:m}}} \quad \mathbf{\Lambda_{pp_{m,1:m}}}]$, 
and $\mathbf{C}'$ describes the remaining of the original $\mathbf{\Lambda}$ (the non-slash-shaded area): $\mathbf{C}'=\{ \forall \lambda_{ij} \in \mathbf{\Lambda}, \forall \lambda_{ij} \notin \mathbf{B}', \forall \lambda_{ij} \notin \mathbf{E}', \forall \lambda_{ij} \notin \mathbf{E}'^\text{T} \}$.

Schur trick works on this linear system, where $\mathbf{v}'$ refers to permuted noises about $\mathbf{x}_1$. 
The marginalization aims to compute $\Delta \mathbf{x}_{{\mathbf{x}_1 } \notin \mathbf{x}}$.

$$
\begin{bmatrix}
    \mathbf{B}' & \mathbf{E}' \\\\
    \mathbf{E}'^\text{T} & \mathbf{C}'
\end{bmatrix}
\begin{bmatrix}
    \Delta \mathbf{x}_{{\mathbf{x}_1 }} \\\\
    \Delta \mathbf{x}_{{\mathbf{x}_1 } \notin \mathbf{x}}
\end{bmatrix}=
\begin{bmatrix}
    \mathbf{v}'_{{\mathbf{x}_1 }} \\\\
    \mathbf{v}'_{{\mathbf{x}_1 } \notin \mathbf{x}}
\end{bmatrix}
$$

The coefficients for $\Delta \mathbf{x}_{{\mathbf{x}_1 } \notin \mathbf{x}}$ should be

$$
(\mathbf{C}'-\mathbf{E}'^\text{T}\mathbf{B}'^{-1}\mathbf{E}')
\Delta \mathbf{x}_{{\mathbf{x}_1 } \notin \mathbf{x}}=
\mathbf{v}'_{{\mathbf{x}_1 } \notin \mathbf{x}} - \mathbf{E}'^\text{T}\mathbf{B}'^{-1} \mathbf{v}'_{\mathbf{x}_1 }
$$

The coefficient matrix $\mathbf{S}'=\mathbf{C}'-\mathbf{E}'^\text{T}\mathbf{B}'^{-1}\mathbf{E}'$ is not sparse as a result of marginalization that removes $\mathbf{x}_1$. Fill-in refers to the dense matrix $\mathbf{S}'$ that derives from $\begin{bmatrix}    \mathbf{B} & \mathbf{E} \\\\    \mathbf{E}^\text{T} & \mathbf{C}   \end{bmatrix}$ which is a sparse matrix.

Denote $\mathbf{S}' = \begin{bmatrix}    \mathbf{B}_{\mathbf{S}'} & \mathbf{E}_{\mathbf{S}'}  \\\\    \mathbf{E}_{\mathbf{S}'} ^\text{T} & \mathbf{C}_{\mathbf{S}'}    \end{bmatrix}$, now the linear system without $\mathbf{x}_1$ can be expressed as

$$
\begin{bmatrix}    
    \mathbf{B}_{\mathbf{S}'} & \mathbf{E}_{\mathbf{S}'}  
    \\\\    
    \mathbf{E}_{\mathbf{S}'} ^\text{T} & \mathbf{C}_{\mathbf{S}'}    
\end{bmatrix}
\begin{bmatrix}
    \Delta \mathbf{x}_{\mathbf{\xi}_{\mathbf{x}_1 \notin \mathbf{x}}} \\\\
    \Delta \mathbf{x}_{\mathbf{p}_{\mathbf{x}_1 \notin \mathbf{x}}}
\end{bmatrix}=
\begin{bmatrix}
    \mathbf{v}_{\mathbf{x}_1 \notin \mathbf{x}} \\\\
    \mathbf{w}_{\mathbf{x}_1 \notin \mathbf{x}}
\end{bmatrix}
$$

Repeat this marginalization process for many more $\mathbf{x}_k$, $\mathbf{S}'_k$ can be very dense.
A dense matrix can be time-consuming in finding the solution.

## Pose Graph Optimization

Computation for landmark positions can be costly as one photo shot can contain thousands of visual features, and as revealed in the sliding window method that repeated adding/removing camera pose $\mathbf{x}_k$ can make the linear system $\mathbf{S}_k$ very dense.
However, once landmark positions are determined (landmark optimization convergent to a small error), unlike a camera's pose, landmarks barely move.

Pose graph agrees on this assumption that mainly focuses on optimizing camera poses.

### Residuals and Jacobians

Define a camera pose $[\mathbf{R}|\mathbf{t}]_i$, for $j \ne i$, define another pose $[\mathbf{R}|\mathbf{t}]_j$ (NOT necessarily the next step $j=1,2, ..., i-2,i-1,i+1,i+2,...$). The movement in between is denoted as $\Delta [\mathbf{R}|\mathbf{t}]_{ij}$

$$
\Delta \mathbf{\xi}_{ij} = 
\mathbf{\xi}_{i}^{-1} \circ \mathbf{\xi}_{j}=
ln([\mathbf{R}|\mathbf{t}]_{i}^{-1} [\mathbf{R}|\mathbf{t}]_{j})^\vee
$$

Similarly in $SE(3)$, there is

$$
[\mathbf{R}|\mathbf{t}]_{ij} = 
[\mathbf{R}|\mathbf{t}]^{-1}_i [\mathbf{R}|\mathbf{t}]_j
$$

The error $\mathbf{e}_{ij}$ that concerns the differences between the ideal pose transformation $[\mathbf{R}|\mathbf{t}]_{ij}$ and the two-pose-based computed transformation $[\mathbf{R}|\mathbf{t}]^{-1}_i [\mathbf{R}|\mathbf{t}]_j$ is defined as

$$
\mathbf{e}_{ij} = 
ln([\mathbf{R}|\mathbf{t}]_{ij}^{-1}[\mathbf{R}|\mathbf{t}]^{-1}_i [\mathbf{R}|\mathbf{t}]_j)
$$

Apply Lie algebra perturbation $\Delta \mathbf{\xi}$ for finding the Jacobian of $\mathbf{e}$.
Since there are $\mathbf{\xi}_i$ and $\mathbf{\xi}_j$, the Jacobian should respect these two variables. Define two trivial disturbance terms $\Delta \mathbf{\xi}_i$ and $\Delta \mathbf{\xi}_j$ to the above error expression.

$$
\hat{\mathbf{e}}_{ij} = 
ln([\mathbf{R}|\mathbf{t}]_{ij}^{-1}[\mathbf{R}|\mathbf{t}]^{-1}_i e^{(-\Delta \mathbf{\xi}_i)^\wedge} e^{(\Delta \mathbf{\xi}_j)^\wedge} [\mathbf{R}|\mathbf{t}]_j)^\vee
$$