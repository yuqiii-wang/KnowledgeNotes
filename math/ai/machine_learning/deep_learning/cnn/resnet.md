# Residual neural network (ResNet)

## Forward propogation

$$
a^l=g(W^{l-1,l} \cdot a_{l-1} + b^l + W^{l-2,l} \cdot a^{l-1})
$$
where $a^l$ is the activation output at the layer $l$; $g$ is the activation function;  $W^{l-1,l}$ is the weight matrix mapping from the $l-1$-th layer to the $l$-th layer.

## Backward propogation

$$
\Delta w^{l-1,l}
=
- \eta \frac{\partial E^l}{\partial w^{l-1,l}}
=-\eta a^{l-1} \cdot \delta^l
\\
\Delta w^{l-2,l}
=
- \eta \frac{\partial E^l}{\partial w^{l-2,l}}
=-\eta a^{l-2} \cdot \delta^l
$$
where $\eta$ is the learning rate; $a^l$ and $\delta^l$ refer to activation and error at the $l$-th layer, respectively.

## Advantages

The good result by ResNet is attributed to its skipping layers that pass error from upper layers to lower layers. This mechanism successfully retains error updating lower layer neurons, different from traditional neural networks that when going too deep, suffers from vanishing gradient issues (lower layer neurons only see very small gradient).