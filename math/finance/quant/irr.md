# Internal Rate of Return (IRR)

IRR is a discount rate that makes the net present value (NPV) of all cash flows equal to zero in a discounted cash flow analysis.

IRR can be used to compute risks for mark-to-market (MTM) security price data.

The most typical form is defined as below.

$$
0 = NPV =
\sum^T_t \frac{C_t}{(1+r)^t} - C_0
$$

where

* $C_t$ cash flow at the $t$ time
* $C_0$ cash flow at the beginning
* $r$ discount rate/IRR
* $t$ the number of time periods

In practice, there are linear/exponential single/multiple instrument forms of the IRR (the above typical form is single exponential IRR formula).

* $n$: grid point of period with $n \in \{ \text{o}/\text{n} (\text{overnight}), \text{t}/\text{n} (\text{tomorrow}), 2 \text{ days}, 3 \text{ days}, 5 \text{ days}, 1 \text{ week}, 2 \text{ weeks}, ..., 30 \text{ years} \}$
* $r_n$ interest rate of instrument for period $n$
* $t_n$ the length of period $i$ of coupon $cp$ of the instrument covering period $n$
* $df_n$ discount factor of period $n$
* $df_{sn}$ discount factor at the start date of yield curve instrument for period $n$

$$\begin{align*}
\text{linear compounding for single coupon instrument} && df_n &= \frac{df_{sn}}{1+r_n t_n} \\
\text{exponential compounding for single coupon instrument} && df_n &= \frac{df_{sn}}{(1+r_n)^{t_n}} \\
\text{linear compounding for multi coupon instruments} && df_n &= \frac{df_{sn}}{(1+r_n)^{t_n}} \\
\end{align*}
$$
