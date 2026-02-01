# Decision Tree

A decision tree tests inputs and branches to other conditions to conduct more tests, till reaching leaf nodes.

It attempts to find conditions with large value uncertainty/entropy and splits into branches to lower uncertainty, such as by Gini Impurity or Information Gain.

||Gini Impurity|Information Gain (Entropy)|
|-|-|-|
|Formula|$G=1-\sum^{C}\_{i=1}p^2_i$|$H=-\sum^{C}\_{i=1}p_i\log_2 p_i$|
|Computational Cost|Faster (no logarithms)|Slower (requires logarithms)|
|Sensitivity|Less sensitive to class probabilities|More sensitive (penalizes mixed nodes more)|

## Gini Impurity

Gini Impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.

Given $C$ classes, Gini Impurity is defined as

$$
G=1-\sum^{C}\_{i=1}p^2_i
$$

## Information Gain

Information gain is based on the concept of entropy and information content from information theory.

For $C$ classes, Information Gain (Entropy) is defined as

$$
IG=H_{\text{parent}}-\sum_{j=1}^k\frac{N_j}{N}H_{\text{child}}
$$

where $N_j$ is samples in child node $j$, and $N$ is total samples.
Entropy $H$ is defined as below

$$
H=-\sum^{C}\_{i=1}p_i\log_2 p_i
$$
