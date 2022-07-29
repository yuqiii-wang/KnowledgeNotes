# Inverse Matrix

A spare matrix $A$ has its inverse when its determiant is not zero.
$$
AA^{-1} - I
$$

and,
$$
A^{-1} = \frac{1}{|A|}Adj(A)
$$
where
$|A|$ is determiant of $A$ and $Adj(A)$ is an adjugate matrix of $A$.

Geometrically speaking, an inverse matrix $A^{-1}$ takes a transformation $A$ back to its origin (same as reseting basis vectors).

## Pseudo inverse

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