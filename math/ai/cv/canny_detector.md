# Canny Detector

reference: http://www.pages.drexel.edu/~nk752/cannyTut2.html

**Apply a Gaussian blur**

A Gaussian blur is applied. E.g., a 5 &times; 5 Gaussian blur matrix is
$$
\begin{bmatrix}
    2/159 & 4/159 & 5/159 & 4/159 & 2/159 \\
    4/159 & 9/159 & 12/159 & 9/159 & 4/159 \\
    5/159 & 12/159 & 15/159 & 12/159 & 5/159 \\
    4/159 & 9/159 & 12/159 & 9/159 & 4/159 \\
    2/159 & 4/159 & 5/159 & 4/159 & 2/159
\end{bmatrix}
$$

**Find edge gradient strength and direction**

The next step is to use Sobel masks to find the edge gradient strength and direction for each pixel.

Sobel operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical. The process goes as below (an example)

$$
G_x=
\begin{bmatrix}
+1 & 0 & -1 \\
+2 & 0 & -2 \\
+1 & 0 & -1
\end{bmatrix}
$$

$$
G_y=
\begin{bmatrix}
+1 & +2 & +1 \\
0 & 0 & 0 \\
-1 & -2 & -1
\end{bmatrix}
$$

Then
$$
G = \sqrt{G_x^2 + G_y^2}
\\
\Theta = atan(\frac{G_x}{G_y})
$$

```cpp
int edgeDir[maxRow][maxCol];			
float gradient[maxRow][maxCol];		

for (row = 1; row < H-1; row++) {
    for (col = 1; col < W-1; col++) {
        gradient[row][col] = sqrt(pow(Gx,2.0) + pow(Gy,2.0));	// Calculate gradient strength			
        thisAngle = (atan2(Gx,Gy)/3.14159) * 180.0;		// Calculate actual direction of edge
        
        /* Convert actual edge direction to approximate value */
        if ( ( (thisAngle < 22.5) && (thisAngle > -22.5) ) || (thisAngle > 157.5) || (thisAngle < -157.5) )
            newAngle = 0;
        if ( ( (thisAngle > 22.5) && (thisAngle < 67.5) ) || ( (thisAngle < -112.5) && (thisAngle > -157.5) ) )
            newAngle = 45;
        if ( ( (thisAngle > 67.5) && (thisAngle < 112.5) ) || ( (thisAngle < -67.5) && (thisAngle > -112.5) ) )
            newAngle = 90;
        if ( ( (thisAngle > 112.5) && (thisAngle < 157.5) ) || ( (thisAngle < -22.5) && (thisAngle > -67.5) ) )
            newAngle = 135;
            
        edgeDir[row][col] = newAngle;
    }
}	
```

**Trace along the edges**

The next step is to actually trace along the edges based on the previously calculated gradient strengths and edge directions.

 If the current pixel has a gradient strength greater than the defined upperThreshold, then a switch is executed. The switch is determined by the edge direction of the current pixel. It stores the row and column of the next possible pixel in that direction and then tests the edge direction and gradient strength of that pixel. If it has the same edge direction and a gradient strength greater than the lower threshold, that pixel is set to white and the next pixel along that edge is tested. In this manner any significantly sharp edge is detected and set to white while all other pixels are set to black.

 The pseduocode below summarizes this process.

 ```cpp
void findEdge(int rowShift, int colShift, int row, int col, int dir, int lowerThreshold){
    int newCol = col + colShift;
    int newRow = row + rowShift;
    while ( (edgeDir[newRow][newCol]==dir) && !edgeEnd && (gradient[newRow][newCol] > lowerThreshold) ) {
        newCol = col + colShift;
        newRow = row + rowShift;
        image[newRow][newCol] = 255; // white, indicates an edge
    }
}

for (int row = 1; row < H - 1; row++) {
	for (int col = 1; col < W - 1; col++) {
        if (gradient[row][col] > upperThreshold) {
            switch (edgeDir[row][col]){		
                case 0:
                    findEdge(0, 1, row, col, 0, lowerThreshold);
                    break;
                case 45:
                    findEdge(1, 1, row, col, 45, lowerThreshold);
                    break;
                case 90:
                    findEdge(1, 0, row, col, 90, lowerThreshold);
                    break;
                case 135:
                    findEdge(1, -1, row, col, 135, lowerThreshold);
                    break;
                default :
                    image[row][col] = 0; // black
                    break;
            }
        }
    }
}
 ```

**Suppress non-maximum edges**

The last step is to find weak edges that are parallel to strong edges and eliminate them. This is accomplished by examining the pixels perpendicular to a particular edge pixel, and eliminating the non-maximum edges.

```cpp
// This function suppressNonMax(...) is called similar to the edge tracing stage where suppressNonMax(...) starts at different edge angles.
void suppressNonMax(int rowShift, int colShift, int row, int col, int dir, int lowerThreshold){
    int newCol = col + colShift;
    int newRow = row + rowShift;
    while ( (edgeDir[newRow][newCol]==dir) && !edgeEnd && (gradient[newRow][newCol] > lowerThreshold) ) {
        nonMax[newRow][newCol] = 0;
        newCol = col + colShift;
        newRow = row + rowShift;
    }
}
```