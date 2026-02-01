# Powell's Conjugate Direction Method

Powell’s method is based on a model quadratic objective function and conjugate
directions in $\mathbb{R}^n$ with respect to the Hessian of the quadratic objective function.

Define *mutually orthogonal*: given two mutually orthogonal vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{n}$, the scalar product {\langlehere denote ${\langle\space.\space\rangle}$ as the scalar product operator\rangle} should be zero ${\langle\mathbf{u}, \mathbf{v}\rangle}=\mathbf{u}^\text{T} \mathbf{v}=0$.

Define *mutually conjugate*: given two mutually conjugate vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{n}$, then $\mathbf{u}$ and $\mathbf{v}$ are said to be mutually conjugate with respect to a symmetric positive definite matrix $A$, if $\mathbf{u}$ and $A\mathbf{v}$ are mutually orthogonal such as ${\langle\mathbf{u}, A\mathbf{v}\rangle}=\mathbf{u}^\text{T} A\mathbf{v}=0$.

Define *eigenvectors* $\mathbf{x}\_i$ corresponding to an *eigenvalue*: 
$$
A \mathbf{x}\_i = \lambda_i \mathbf{x}\_i
$$
If $A \in \mathbb{R}^{n \times n}$ is a symmetric positive definite matrix, then there will exist $n$ eigenvectors, $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n$ which are mutually orthogonal ${\langle\mathbf{x}\_i, \mathbf{x}_j\rangle}=0$.

Since ${\langle\mathbf{x}\_i, A\mathbf{x}_j\rangle}={\langle\mathbf{x}\_i, \lambda_j\mathbf{x}_j\rangle}=0, \forall i \ne j$, this implies that the
eigenvectors, $\mathbf{x}\_i$, are mutually conjugate with respect to the matrix $A$.