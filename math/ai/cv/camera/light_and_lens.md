# Light and Lens

Remember how light passes through a len.

![light_and_lens](imgs/light_and_lens.png "light_and_lens")

## Lambertian reflectance

Lambertian reflectance is the property that defines an ideal "matte" or diffusely reflecting surface. Surface luminous intensity obeys Lambert's cosine law.

Ideal "matte" surface's luminance is isotropic, while specular reflection sees high intensity in one direction.

![diffuse_reflection](imgs/diffuse_reflection.png "diffuse_reflection")


## Pinhole

A pinhole camera is a simple camera without a lens and with a single small aperture.

![pinhole_img_transform](imgs/pinhole_img_transform.png "pinhole_img_transform")

### Intrinsics

Define Intrinsic Parameters
$$
K=
\begin{bmatrix}
      f_x & 0 & 0 \\
      s & f_y & 0 \\
      c_x & c_y & 1
\end{bmatrix}
$$
where 
* $(c_x, c_y)$ Optical center (the principal point), in pixels.
* $F$ focal length in world units, often in millimeters
* $(f_x, f_y)$ focal length in pixels, where $f_x=F/p_x$, $f_y=F/p_y$, in which $p_x \times p_y$ are the size of an image

### Pinhole Intrinsic Matrix $K$ Derivation

Define image pixel $(x_p, y_p)$, camera translation of the origin $(c_x, c_y)$ from the real world point $(x_w, y_w)$, and two scaling factor $\alpha, \beta$ the mapping relationship is
$$
\begin{align*}
x_p &= \alpha x_w +c_x
\\
y_p &= \beta y_w +c_y
\end{align*}
$$

Consider the symmetric mapping around the origin $O$ indicated as below.

![pinhole_1d_projection](imgs/pinhole_1d_projection.png "pinhole_1d_projection")

The world-to-image mapping can be expressed as
$$
\begin{align*}
x_p &= \frac{f_x}{Z} x_w +c_x
\\
y_p &= \frac{f_y}{Z} y_w +c_y
\end{align*}
$$

So that
$$
\begin{bmatrix}
      x_p \\
      y_p \\
      1
\end{bmatrix}=
\frac{1}{Z}
\begin{bmatrix}
      f_x & 0 & c_x \\
      0 & f_y & c_y \\
      0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
      x_w \\
      y_w \\
      Z
\end{bmatrix}
\overset{\Delta}{=}
\frac{1}{Z} K^\text{T} \bold{p}
$$

Here inserts $s$ indicating axis-skew, and derives
$$
K=
\begin{bmatrix}
      f_x & 0 & 0 \\
      s & f_y & 0 \\
      c_x & c_y & 1
\end{bmatrix}
$$

### Extrinsics

Define Extrinsic Parameters
$$
\begin{align*}
\begin{bmatrix}
      \bold{R} \\
      \bold{t}
\end{bmatrix}_{4 \times 4}
& =
\begin{bmatrix}
      r_{11} & r_{12} & r_{13} & t_{1} \\
      r_{21} & r_{22} & r_{23} & t_{2} \\
      r_{31} & r_{32} & r_{33} & t_{3} \\
      0 & 0 & 0 & 1
\end{bmatrix}
\\
\begin{bmatrix}
      \bold{R} \\
      \bold{t}
\end{bmatrix}_{4 \times 3}
& =
\begin{bmatrix}
      r_{11} & r_{12} & t_{1} \\
      r_{21} & r_{22} & t_{2} \\
      r_{31} & r_{32} & t_{3} \\
      0 & 0 & 1
\end{bmatrix}
\end{align*}
$$
where $R$ and $\bold{t}$ are rotation and translation matrices that adjust a camera physical orientation and lateral position. Here we set $z=0$ to assume that both real world and camera frame are parallel to each other (no rotation but possible translation).

The mapping relationship from world points to image pixels can be expressed as
$$
w
\begin{bmatrix}
      x_p \\
      y_p \\
      1
\end{bmatrix}^\text{T}=
\begin{bmatrix}
      w x_w \\
      w y_w \\
      w \\
      1
\end{bmatrix}^\text{T}
\begin{bmatrix}
      \bold{R} \\
      \bold{t}
\end{bmatrix}
_{4 \times 3}
\begin{bmatrix}
      f_x & 0 & 0 \\
      s & f_y & 0 \\
      c_x & c_y & 1
\end{bmatrix}
$$
where $(x_p, y_p)$ is image pixel. $(w x_w, w y_w, w)$ is world point pixel. $w$ is the scaling factor (same as the $Z$ in intrinsics). Usually $w=1$.

## Fisheye

### Distortion Formulation

* Radial distortion

Radial distortion occurs when light rays bend more near the edges of a lens than they do at its optical center. The smaller the lens, the greater the distortion.

![radial_distortion](imgs/radial_distortion.png "radial_distortion")

* Tangent distortion

Tangential distortion occurs when the lens and the image plane are not parallel. 

![tangent_distortion](imgs/tangent_distortion.png "tangent_distortion")

Distortion correction formula:

$$
\begin{align*}
x_u = & \space x_d+(x_d-c_x)(k_1r^2+k_2r^4+k_3r^6+...)
\\ &+ 
\big(p_1(r^2+2(x_d-c_x)^2)+2p_2(x_d-c_x)(y_d-c_y)\big)
(1+p_3r^2+p_4r^4+...)
\end{align*}
\\
\begin{align*}
y_u = & \space y_d+(y_d-c_y)(k_1r^2+k_2r^4+k_3r^6+...)
\\ &+ 
\big(p_1(r^2+2p_2(x_d-c_x)(y_d-c_y))+2p_2(y_d-c_y)^2\big)
(1+p_3r^2+p_4r^4+...)
\end{align*}
$$
where 
* $(x_d, y_d)$ is the distorted image pixel as projected on image plane using specified lens
* $(x_u, y_u)$ is the undistorted image pixel after correction
* $(c_x, c_y)$ is the distortion center
* $k_n$ is the $n$-th radial distortion coefficient
* $p_n$ is the $n$-th tangential distortion coefficient
* $r=\sqrt{(x_d-c_x)^2+(y_d-c_y)^2}$ is the Euclidean distance between the distorted image point and the distortion center

Usually, for images of little distortions, $n=2$ can render good recovery result ($k_n$ s of higher $n$ are insignificant that $k_n \approx 0$). Greater the distortion, higher the power factor $n$ is required. In OpenCV, $n=4$ is used.

### Fisheye intrinsic and extrinsic parameters

Remember we have the below camera model. Fisheye distortion adds polynomials so that, define $(x_d, y_d)$ as the distortion pixel coordinate.

$$
w
\begin{bmatrix}
      x_p \\
      y_p \\
      1
\end{bmatrix}^\text{T}=
\begin{bmatrix}
      w x_d \\
      w y_d \\
      w \\
      1
\end{bmatrix}^\text{T}
\begin{bmatrix}
      \bold{R} \\
      \bold{t}
\end{bmatrix}
_{4 \times 3}
\begin{bmatrix}
      f_x & 0 & 0 \\
      s & f_y & 0 \\
      c_x & c_y & 1
\end{bmatrix}
$$

where $(x_d, y_d)$ is the distortion mapping from the real world point $(x_w, y_w)$
$$
w\begin{bmatrix}
      x_d \\
      y_d
\end{bmatrix}=
w(1+k_1r^2+k_2r^4+k_3r^6+k_4r^8)
\begin{bmatrix}
      x_w \\
      y_w
\end{bmatrix}
$$
where
* $(x_w, y_w)$ is a real world point
* $(x_d, y_d)$ is an ideal image projection (retaining fisheye distorted features) of a real-world point
* $w$ is a scaling factor
* $k_0, k_2, k_3, k_4$ are polynomial coefficients
* $r=\sqrt{(x_d-c_x)^2+(y_d-c_y)^2}$ is the distance of a point from the image center

Consider tangent distortion for tilted planes, $(x_d, y_d)$ can have the below expression:
$$
w\begin{bmatrix}
      x_d \\
      y_d
\end{bmatrix}=
\begin{bmatrix}
      x_w \\
      y_w
\end{bmatrix}
+
\begin{bmatrix}
      2p_1x_wy_w+p_2(r^2+2x_w^2) \\
      2p_2x_wy_w+p_1(r^2+2y_w^2)
\end{bmatrix}
$$
where $p_1, p_2$ are tangent distortion coefficients.

## Triangulation and Parallax

Triangulation is an indirect method of measuring the distance to an object by imagining a triangle between the observer at two positions and the object. 
Parallax, as used in this activity, is the apparent motion of a relatively close object when viewed by the observer from two different positions.

## Viewing Frustum

## Exposure Triangle: Sensor ISO, Aperture and Shutter Speed 


<div style="display: flex; justify-content: center;">
      <img src="imgs/iso_aperture_shutter_triangle.png" width="40%" height="20%" alt="iso_aperture_shutter_triangle" />
</div>
</br>

* ISO

ISO is an international standard describing the sensitivity to light as pertains to either film or a digital sensor. 

It affects image granularity such as noises and brightness.

A large ISO value should be used at night time or in cloudy weather (ISO $> 500$);
a small ISO value should be used on a sunny day (ISO $<100$)

* Aperture

Aperture is a hole or an opening through which light travels. 
It is expressed as $\text{f}/\text{stop} = \text{focal length}/\text{diameter of aperture}$

It affects focal length.

* Shutter Speed

Shutter is a device that allows light to pass for a determined period.

Shutter Speed says about how fast the shutter is closed.
It is expressed as fractions of a second, so $1/50$ means $20$ ms.

Slow shutter speed renders image blurred for moving objects.
Slow speed allows more light come into camera, resulted in over-exposure (this can be re-balanced by histogram equalization).