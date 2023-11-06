# Chi-Squared Distribution and Tests

## Chi-Squared Distribution

Chi-squared distribution (also chi-square or $X^2$-distribution) with $k$ degrees of freedom is the distribution of a sum of the squares of $k$ independent standard normal random variables. 

If $Z_1, ..., Z_k$ are independent, standard normal random variables, then the sum of their squares,
$$
Q = \sum^k_{i=1} Z^2_i 
$$

It is often denoted as $Q \sim X^2(k)$ or  $Q \sim X^2_k$

### Probability Density Function (PDF)

For a positive $x$ and positive integer $k$, the value

$$
f_{X^2}(x,k)=
\left\{
    \begin{array}{c}
        \frac{x^{\frac{k}{2}-1}e^{\frac{k}{2}}}{2^{\frac{k}{2}}\Gamma(\frac{k}{2})} & x>0, k \in \mathbb{Z}
        \\
        \space
        \\
        0 & \text{otherwise}
    \end{array}
\right.
$$
where $\Gamma$ is a Gamma function given a positive integer $k$, there is $\Gamma(k)=(k-1)!$

### Chi-Square Distribution Table

Below is a pre-computed Chi-square distribution table. 
For example, the number $18.548$ represents that the integral (marked as the shaded grey area in the figure below) is $\int_{18.548}^{+\infty} f_{X^2} = 0.005$, given the degree of freedom $k=6$.

![chi_square_dist_table](imgs/chi_square_dist_table.png "chi_square_dist_table")


## Chi-Squared Tests

Chi-squared test is used to determine whether there is a statistically significant difference between the expected frequencies and the observed frequencies in one or more categories of a contingency table.

### Example

A city has a population of $1,000,000$ living in four districts: $A,B,C,D$. $650$ citizens are surveyed for their occupations, labelled as "white collar", "blue collar", or "no collar".

The surveyed results are shown as below.

||$A$|$B$|$C$|$D$|Total|
|-|-|-|-|-|-|
|White collar|$90$|$60$|$104$|$95$|$349$|
|Blue collar|$30$|$50$|$51$|$20$|$151$|
|No collar|$30$|$40$|$45$|$35$|$150$|
|Total|$150$|$150$|$200$|$150$|$650$|

For a total of $349$ out of $650$ people reported to have white collar jobs city-wise, and for district $A$ there are $90$ out of $150$, there is

$$
150 \times \frac{349}{650} \approx 80.54
$$

then

$$
\frac{(observed-expected)^2}{expected} =
\frac{(90-80.54)^2}{80.54}
\approx 1.11
$$

This is one frequency for $A$ district people having white collar jobs. In total, there are $3 \times 4 = 12$ frequencies.

Chi-squared $X^2$ sums up all frequencies, there is

$$
\begin{align*}
X^2 &= \sum \frac{(observed-expected)^2}{expected}
\\ &=
\frac{(90-150\times\frac{349}{650})^2}{150\times\frac{349}{650}}
+
\frac{(30-150\times\frac{151}{650})^2}{150\times\frac{151}{650}}
+
\frac{(30-150\times\frac{150}{650})^2}{150\times\frac{150}{650}}
\\
& \quad +
\frac{(60-150\times\frac{349}{650})^2}{150\times\frac{349}{650}}
+
\frac{(50-150\times\frac{151}{650})^2}{150\times\frac{151}{650}}
+ ...
\\ &\approx
1.11 + 0.672 + 0.672 + 5.23 + 6.59 + ...
\\ &\approx 24.57
\end{align*}
$$

The null hypothesis is that each person's neighborhood of residence is independent of the person's occupational classification. 
Under the null hypothesis, this sum has approximately a chi-squared distribution whose number of degrees of freedom $k$ is
$$
(NumberOfRows-1) \times (NumberOfCols-1) 
=
(3-1)\times(4-1)=6
$$

Given the aforementioned Chi-square distribution table, $24.57 > 18.548$ represents the confidence is greater than $1-0.005=0.995$ for the belief that the null hypothesis is false. In other words, occupations and residential districts are correlated.