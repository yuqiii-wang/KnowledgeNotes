# Gram–Schmidt process

Gram–Schmidt process is a method for orthonormalizing a set of vectors in an inner product space.

Denote a projection operator from vector $\bold{v}$ onto $\bold{u}$:
$$
proj_{\bold{u}}(\bold{v})
=
\frac{<\bold{u},\bold{v}>}{<\bold{u},\bold{u}>}\bold{u}
$$
where $<\bold{u},\bold{v}>$ represents inner product operation.

$$
\begin{array}{cc}
    \bold{u}_1 = \bold{v}_1 & 
    \bold{e}_1=\frac{\bold{u}_1}{||\bold{u}_1||}
    \\
    \bold{u}_2 = \bold{v}_2 - proj_{\bold{u}_1}(\bold{v}_2) & 
    \bold{e}_2=\frac{\bold{u}_2}{||\bold{u}_2||}
    \\
    \bold{u}_3 = \bold{v}_3 - proj_{\bold{u}_1}(\bold{v}_3) - proj_{\bold{u}_2}(\bold{v}_3) & 
    \bold{e}_3=\frac{\bold{u}_3}{||\bold{u}_3||}
    \\
    \bold{u}_4 = \bold{v}_4 - proj_{\bold{u}_1}(\bold{v}_4) - proj_{\bold{u}_2}(\bold{v}_4) - proj_{\bold{u}_3}(\bold{v}_4) & 
    \bold{e}_4=\frac{\bold{u}_4}{||\bold{u}_4||}
    \\
    \space
    \\
    ... & ...
    \\
    \space
    \\\
    \bold{u}_k = \bold{v}_k - \sum^{k-1}_{j}proj_{\bold{u}_j}(\bold{v}_k) &
    \bold{e}_k=\frac{\bold{u}_k}{||\bold{u}_k||}
\end{array}
$$

![Gram–Schmidt_process.svg](imgs/Gram–Schmidt_process.svg.png "Gram–Schmidt_process.svg")
