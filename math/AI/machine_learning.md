# Machine Learning 

## Loss/Cost Func

A loss function is for a single training example. A cost function, on the other hand, is the average loss over the entire training dataset. 

### **Regression**

### Squared Error Loss

Squared Error loss for each training example, also known as L2 Loss, and the corresponding cost function is the Mean of these Squared Errors (MSE).

<center>
L = (y - f(x))<sup>2</sup>
</center>

### Absolute Error Loss

Also known as L1 loss. The cost is the Mean of these Absolute Errors (MAE).

<center>
L = | y - f(x) |
</center>

### Huber Loss

The Huber loss combines the best properties of MSE and MAE. It is quadratic for smaller errors and is linear otherwise (and similarly for its gradient). It is identified by its delta parameter:

<center>
L<sub>&delta;</sub> = {
</center>    
<center>
1/2 (y - f(x))<sup>2</sup>, if |y - f(x)| < &delta;
</center>
<center>
&delta;|y - f(x)| - 1/2 &delta;<sup>2</sup>, otherwise
</center>

### **Classification**



## Algos

### Ensemble Methods
 Ensemble methods are learning algorithms that construct a set of classifiers/regressors and then classify/regress new data points by taking a weighted vote of their predictions.

### AdaBoost
Problems in machine learning often suffer from the curse of dimensionality, and to overcome this issue, the AdaBoost training process selects only those features known to improve the predictive power of the model, reducing dimensionality and potentially improving execution time as irrelevant features need not be computed.
A plain explanation is that this methodology adjusts weights of samples so that classifiers/regressors have different votes (in contrast to random forests having equal says over each tree's result).

### Auto Encoder/Decoder
Used NN to encode/decode input. It demonstrates its capability to "remember" the input and re-render it.

### Generative Adversarial Networks (GANs) 
Given a training set, this technique learns to generate new data with the same statistics as the training set by use of a discriminator and a generator competing against each other. 
Generator takes noises as input and feed its output to a discriminator and the discriminator will compute the loss judging validity. The loss will be backprobagated to the generator so that the generator will learn the differences. When the generator is good enough to imitate the true input the discriminator starts to get confused about validity. Since the loss is backpropagated to discriminator as well, discriminator will start to learn how to distinguish the forged input.

### Markov Chain