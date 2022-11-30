# Powell's Method

Powell’s method is based on a model quadratic objective function and conjugate
directions in $\mathbb{R}^n$ with respect to the Hessian of the quadratic objective function.

Define *mutually orthogonal*: given two mutually orthogonal vectors $\bold{u}, \bold{v} \in \mathbb{R}^{n}$, the scalar product (here denote $(.)$ as the scalar product operator) should be zero $(\bold{u}, \bold{v})=\bold{u}^\text{T} \bold{v}=0$.

Define *mutually conjugate*: given two mutually conjugate vectors $\bold{u}, \bold{v} \in \mathbb{R}^{n}$, then $\bold{u}$ and $\bold{v}$ are said to be mutually conjugate with respect to a symmetric positive definite matrix $A$, if $\bold{u}$ and $A\bold{v}$ are mutually orthogonal such as $(\bold{u}, A\bold{v})=\bold{u}^\text{T} A\bold{v}=0$.

Define *eigenvectors* $\bold{x}_i$ corresponding to an *eigenvalue*: 
$$
A \bold{x}_i = \lambda_i \bold{x}_i
$$
If $A \in \mathbb{R}^{n \times n}$ is a symmetric positive definite matrix, then there will exist $n$ eigenvectors, $\bold{x}_1, \bold{x}_2, ..., \bold{x}_n$ which are mutually orthogonal $(\bold{x}_i, \bold{x}_j)=0$.

Since $(\bold{x}_i, A\bold{x}_j)=(\bold{x}_i, \lambda_j\bold{x}_j)=0, \forall i \ne j$, all eigenvectors of $A$ are mutually conjugate.