# Some Best Practice Discussions

## Gradient Explosion/Vanishing

A typical layer of neural network has a matrix multiplication operation and an activation function mapping $h_{k+1}=\sigma(W h_k)$,
whose derivative is $\frac{\partial h_{k+1}}{\partial h_k}=\text{diag}\big( \sigma'(W h_k) \big) W$ (for $\sigma(.)$ is a 2d to 1d mapping function, its derivative can be represented by $\text{diag}(.)$ ).

The determinant of $\text{diag}\big( \sigma'(W h_k) \big)$ is smaller than $1$, and the determinant of $W$ varies (could be greater/less than $1$).
Hence, the total result $\frac{\partial h_{k+1}}{\partial h_k}=\text{diag}\big( \sigma'(W h_k) \big) W$ could be greater/less than $1$, and if $\frac{\partial h_{k+1}}{\partial h_k}$ cannot stay around $1$, there is a risk of gradient explosion/vanishing.

## Overfitting

* Use *Dropout* to enhance robustness of a network 

### For CV

* Random Image Crop
* Random Flip/Rotation

## Optimization Tuning

* Use Adam

## Saddle Point Escape