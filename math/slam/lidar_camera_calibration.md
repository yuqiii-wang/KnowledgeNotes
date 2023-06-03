# Lidar Camera Calibration

Lidar-camera calibration is about finding the right transform between camera and lidar.
The calibration process is usually done through matching against chessboards.

<div style="display: flex; justify-content: center;">
      <img src="imgs/lidar_camera_extrinsic.png" width="30%" height="20%" alt="lidar_camera_extrinsic" />
</div>
</br>

<div style="display: flex; justify-content: center;">
    <img src="imgs/lidar_camera_cali_chessboard.png" width="60%" height="40%" alt="lidar_camera_cali_chessboard" />
</div>
</br>


Ideally, the lidar and camera outputs would fuse into one by the calibrated extrinsic transform.

<div style="display: flex; justify-content: center;">
    <img src="imgs/lidar_cam_calibration.png" width="60%" height="40%" alt="lidar_cam_calibration" />
</div>
</br>

