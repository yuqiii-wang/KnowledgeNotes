# Basic machine learning knowledge

## Cost vs Loss

The cost function measures the model's error on a group of objects, whereas the loss function deals with a single data instance.

## Supervision and Non-Supervision Learning

* Supervision

* Semi-Supervised Learning

Semi-supervised learning is a special instance of weak supervision that combines a small amount of labeled data with a large amount of unlabeled data during training.

* Non-Supervision Learning

## Ablation

Ablation is the removal of a component of an AI system.
An ablation study investigates the performance of an AI system by removing certain components to understand the contribution of the component to the overall system.

Examples:

* Replace a layer of a deep network and test it

## Partial Derivative: With Respect to Weight vs Input

* Weight
$\frac{\partial L}{\partial \bold{w}}$: Directly affects the learning process by adjusting the weights.
* Input
$\frac{\partial L}{\partial \bold{x}}$: Determines how error signals are distributed across the network.
