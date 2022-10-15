# ORB (Oriented FAST and Rotated BRIEF)

## Fast(Features from Accelerated and Segments Test)

Given a pixel $p$ in an array fast compares the brightness of $p$ to surrounding $16$ pixels that are in a small circle around $p$. Pixels in the circle is then sorted into three classes (lighter than $p$, darker than $p$ or similar to $p$). If more than $8$ pixels are darker or brighter than $p$ than it is selected as a keypoint.