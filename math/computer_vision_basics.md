Computer Vision Basics

1. SVD for Image Compression

reference: http://web.mit.edu/course/other/be.400/OldFiles/www/SVD/Singular_Value_Decomposition.htm

Assume an image &Chi; = {&Chi;<sub>1</sub>, &Chi;<sub>2</sub>, ..., &Chi;<sub>n</sub>} of a size of n &times; m, with &Chi;<sub>i</sub> consisted of m elements. We want to achieve 
<center>
X = U<sub>n&times;n</sub> S<sub>n&times;m</sub> V<sub>m&times;m</sub><sup>T</sup>
</center>

The eigen-decomposition of X X<sup>T</sup> is
<center>
X X<sup>T</sup> = E&Lambda;E<sup>-1</sup>
</center>
where
E = {e<sub>1</sub>, e<sub>2</sub>, ..., e<sub>n</sub>}
are n eigenvectors of X X<sup>T</sup>, and &Lambda; = diag(&Lambda;<sub>1</sub>, &Lambda;<sub>2</sub>, ..., &Lambda;<sub>n</sub>) containing n eigenvalues. Since X X<sup>T</sup> is symmetric, the eigenvectors are orthogonal and each can be scaled to have unit length, that
<center>
E E<sup>T</sup> = I
</center>
implies
E<sup>-1</sup> = E<sup>T</sup>,
so that
<center>
X X<sup>T</sup> = E&Lambda;E<sup>T</sup>
</center>
This consideration is from the fact that images are often not orthogonal in eigen-decomposition, as the derived eigenvectors should be othogonal to achieve high compression rate.

The eigenvectors of X<sup>T</sup>X make up the columns of V, the eigenvectors of XX<sup>T</sup> make up the columns of U. ||V|| = 1 and ||U|| = 1 are defined to make U and V unitary. Thus, the singular values in S are square roots of eigenvalues from X<sup>T</sup>X or XX<sup>T</sup>. 

Eigenvalues in S are ranked by value to select eigenvectors with most information.

Another usage of SVD for image compression is that there are n images with m pixels. Rearrange all images we have a matrix with a size of n &times; m, then continue the aforementioned process.

P.S. Proof:

X=USV<sup>T</sup> and X<sup>T</sup>=VSU<sup>T</sup>

X<sup>T</sup>X = VSU<sup>T</sup>USV<sup>T</sup>

X<sup>T</sup>A = VS<sup>2</sup>V<sup>T</sup>

X<sup>T</sup>XV = VS<sup>2</sup>

2. Canny Detector

reference: http://www.pages.drexel.edu/~nk752/cannyTut2.html

**Apply a Gaussian blur**

A Gaussian blur is applied. E.g., a 5 &times; 5 Gaussian blur matrix is
<center>

2/159 , 4/159, 5/159, 4/159, 2/159

4/159 , 9/159, 12/159, 9/159, 4/159

5/159 , 12/159, 15/159, 12/159, 5/159

4/159 , 9/159, 12/159, 9/159, 4/159

2/159 , 4/159, 5/159, 4/159, 2/159
</center>

**Find edge gradient strength and direction**

The next step is to use Sobel masks to find the edge gradient strength and direction for each pixel.

Sobel operator uses two 3×3 kernels which are convolved with the original image to calculate approximations of the derivatives – one for horizontal changes, and one for vertical. The process goes as below (an example)

G<sub>x</sub> = 
<center>

+1, 0, -1

+2, 0, -2

+1, 0, -1
</center>

G<sub>x</sub> = 
<center>

+1, +2, +1

0, 0, 0

-1, -2, -1
</center>

Then
<center>
G = &radic;G<sub>x</sub><sup>2</sup> + G<sub>y</sub><sup>2</sup>

&Theta; = atan(G<sub>x</sub> / G<sub>y</sub>)
</center>

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