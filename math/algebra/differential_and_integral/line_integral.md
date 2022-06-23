# Line Integral

Line integral is used for vector field function. Intuitively, line integral works on curve that does not flow along with integral direction, but  takes integral with respect to each integral direction.

![line_integral](imgs/line_integral.jpg "line_integral")


## Example

Define a vector field
$$
\overrightarrow{F}(x,y) = P(x,y)\overrightarrow{i} + Q(x,y)\overrightarrow{j}
$$

Define a curve
$$
\overrightarrow{r}(t) = x(t)\overrightarrow{i} + y(t)\overrightarrow{j}
$$

The line integral of 
$\overrightarrow{F}$ along $C$ is
$$
\int_C \overrightarrow{F} \cdot d \overrightarrow{r}
=
\int^a_b \overrightarrow{F}(\overrightarrow{r}(t)) \cdot \overrightarrow{r}'(t) dt
$$

where
$$
\overrightarrow{F}(\overrightarrow{r}(t)) = \overrightarrow{F}(x(t), y(t))
$$