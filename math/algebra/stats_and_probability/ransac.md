# Random sample consensus (RANSAC)

Random sample consensus (RANSAC) is an iterative method to estimate model parameters when outliers are to be accorded no influence on the values of the estimates. In other words, it is an outlier detection method.

1. Select a fitting model and some random data points

The chosen model is at the user's discretion, that can be a linear or quadratic function, or a more complex prediction model.

Data point selection is totally random.

2. Apply the fitting model on the sampled random data points

The applied fitting model finds the optimal parameters for configuration by the least squared error.

The inlier data points from the entire dataset falling in this fitting model are recorded.

3. Repeat the 1. and 2. step until the entire dataset is covered/sampled

4. Check the recorded data points collected from the 2. step.

If a data point is viewed by the fitting model as a inlier point multiple times during the repeated 1. and 2. steps, this data point is considered an inlier.