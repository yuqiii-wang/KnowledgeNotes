# Evaluation Metrics

## Confusion Matrix 

<div style="display: flex; justify-content: center;">
      <img src="imgs/confusion_matrix.png" width="40%" height="30%" alt="confusion_matrix.svg" />
</div>
</br>

* accuracy (ACC)

$$
ACC = \frac{TP+TN}{TP+TN+FP+FN}
$$

* sensitivity, recall, hit rate, or true positive rate (TPR)

$$
TPR = \frac{TP}{TP+FN}
$$

* fall-out or false positive rate (FPR)

$$
FPR = \frac{FP}{TP+TN}
$$


* precision or positive predictive value (PPV)

$$
PPV = \frac{TP}{TP+FP}
$$

* F1 score

$$
F_1 = 2 \times  \frac{PPV \times TPR}{PPV + TPR}
$$

* Matthews correlation coefficient (MCC)

$$
MCC= \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$


## Correlations

### Pearson Correlation

*Pearson correlation coefficient* measures linear correlation between two sets of data, defined as the ratio between the covariance of two variables and the product of their standard deviations.

$$
\rho_{X,Y} = 
\frac{\text{cov}(X,Y)}{\sigma_X \sigma_Y} =
\frac{E\big( (X - \overline{X}) \cdot (Y - \overline{Y}) \big)}{\sigma_X \sigma_Y}
$$

where $\cdot$ denotes element-wise multiplication that produces a vector of results that satisfies $\text{assert}\big(\text{len}(X) == \text{len}(Y) == \text{len} \big( (X - \overline{X}) \cdot (Y - \overline{Y}) \big) \big)$,
and $E(...)$ is the expectation.
$\sigma_X$ and $\sigma_Y$ are standard deviations.

|Correlation Value Range|Semantics|
|-|-|
|$[0, 0.3)$|No correlation|
|$[0.3, 0.5)$|Low correlation|
|$[0.5, 0.7)$|Moderately correlated|
|$[0.7, 0.9)$|Highly correlated|
|$[0.9, 1]$|Very highly correlated|

If values are negative, correlation is in a reverse order.

### Spearman Correlation

Pearson correlation assesses how strong the linear relationship is between samples, such that for $x_i \in X$ and $y_i \in Y$, Pearson correlation of $1$ should see equal value increase such that $x_i + \Delta x = y_i + \Delta x$.

Spearman's correlation $r_s \in [-1, 1]$ assesses monotonic (linear or non-linear) relationships that there is always  $x_{i+1} > x_i \in X$ and $y_{i+1} > y_i \in Y$.

For example, the non-linear monotonic increase sigmoid gives a Pearson correlation of $0.88$, but Spearman correlation of $1$, as each $\{ x_i, y_i \}$ pair sees for each $x_{i+1} > x_i$, the corresponding sample point should see $y_{i+1} > y_i$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/spearman_vs_pearson.svg.png" width="25%" height="25%" alt="spearman_vs_pearson.svg" />
</div>
</br>

By definition, the Spearman correlation coefficient $r_s$ is defined as the Pearson correlation coefficient between the rank variables (rank variables refer to for variables with index $i=1,2,...$, there exists either $x_{i+1}>x_i$ or $x_{i+1}<x_i$ for all $i=1,2,...$).

For a sample of size $n$, the n$ $raw scores $X, Y$ are converted to ranks $R(X)$ and $R(Y)$, so that $r_s$ is

$$
r_s = \rho_{R(X), R(Y)} = 
\frac{\text{cov}\big( R(X), R(Y) \big)}{\sigma_{R(x)} \sigma_{R(Y)}}
$$

### Matthews Correlation (Phi Coefficient)

*Matthews correlation* (*Phi coefficient*) $r_{\phi}$ is a measure of association for two binary variables.

Define $n_{xy}$ as the count of binary records with respect to $X$ and $Y$ such that

||y=1|y=0|total|
|-|-|-|-|
|x=1|$n_{11}$|$n_{10}$|$n_{1y}$|
|x=0|$n_{01}$|$n_{00}$|$n_{0y}$|
|total|$n_{x1}$|$n_{x0}$|$n$|

So that $r_{\phi}$ is defined as

$$
r_{\phi}= \frac{n_{11} n_{00} - n_{10} n_{01}}{\sqrt{n_{1y}n_{0y}n_{x1}n_{x0}}}
$$

In machine learning, it can be used to measure confusion matrix

$$
MCC= \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

## Receiver operating characteristic (ROC)

The ROC curve is the plot of the true positive rate ($TPR=\frac{TP}{TP+FP}$) against the false positive rate ($FPR=\frac{FP}{FP+TN}$) at each threshold setting.

<div style="display: flex; justify-content: center;">
      <img src="imgs/roc_curve.svg.png" width="25%" height="25%" alt="roc_curve.svg.png" />
</div>
</br>