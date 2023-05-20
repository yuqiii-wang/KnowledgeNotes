# Inverse Matrix

A square matrix $A$ has its inverse when its determinant is not zero.
$$
AA^{-1} - I
$$

and,
$$
A^{-1} = \frac{1}{|A|}Adj(A)
$$
where
$|A|$ is determiant of $A$ (denoted as $det(A)$) and $Adj(A)$ is an adjugate matrix of $A$.

Geometrically speaking, an inverse matrix $A^{-1}$ takes a transformation $A$ back to its origin (same as reseting basis vectors).

## Pseudo Inverse

Pseudo inverse (aka Moore–Penrose inverse) denoted as $A^{\dagger}$, satisfying the below conditions:

* $AA^{\dagger}$ does not neccessarily give to identity matrix $I$, but mapping to itself
$$
AA^{\dagger}A=A
\\
A^{\dagger}AA^{\dagger}=A^{\dagger}
$$

* $AA^{\dagger}$ is Hermitian, and vice versa
$$
(AA^{\dagger})^*=AA^{\dagger}
\\
(A^{\dagger}A)^*=A^{\dagger}A
$$

* If $A$ is invertible, its pseudoinverse is its inverse
$$
A^{\dagger}=A^{-1}
$$

### Pseudo Inverse for Non-Square Matrix Inverse

Given a non-square matrix $A \in \mathbb{R}^{n \times m}$ for $m \ne n$, the "best approximation" of the inverse is defined as $A^{\dagger}$ that satisfies the above pseudo inverse definition $AA^{\dagger}A=A$.
By strict definition, non-square matrix has no inverse.

The motivation is that, consider a linear system $A\bold{x} = \bold{b}$, if $A$ is a square matrix ($m=n$), there is an exact solution for the system $\bold{x}=A^{-1}\bold{b}$, if not ($m \ne n$), there is $AA^{\dagger}A=A$.

To approximate $\bold{x}=A^{-1}\bold{b}$ for non-square matrix $A$, set $\bold{x}=A^{\dagger}\bold{b}$ as the pseudo solution, so that there is $A\bold{x}=AA^{\dagger}\bold{b}=\bold{b}$, where $A^{\dagger} \in \mathbb{R}^{m \times n}$.

OpenCV has builtin API for $\bold{x}=A^{\dagger}\bold{b}$.
In the below code, first construct the linear system by pushing back rows (such as robot states) to `A` and `b`.
Then, find the pseudo inverse of `A` denoted as `pinA`, by which the solution can be constructed as `x = pinA * b;`.

In least squares problem, solution $\bold{x} \in \mathbb{R}^m$ should be derived from an over-determined system where $n > m$. 

```cpp
double cv::invert	(	InputArray 	src,
                        OutputArray 	dst,
                        int 	flags = DECOMP_LU 
                    );

cv::Mat A;
cv::Mat b;
cv::Mat pinA;
 
for (int i = 0; i < nRows; i++) {
    A.push_back(tempA);
    b.push_back(tempb);
}

cv::invert(A, pinA, DECOMP_SVD);

cv::Mat x = pinA * b;
```