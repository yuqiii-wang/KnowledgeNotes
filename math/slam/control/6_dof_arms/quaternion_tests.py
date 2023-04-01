import numpy as np
import quaternion # pip install numpy-quaternion
import math
 
def computeEulerFromQuaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

# https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
def rotateVecByQuaternion(q, srcVec):

    u = np.array([q.x, q.y, q.z])
    s = q.w

    tmp1 = np.dot(u, srcVec)
    tmp2 = s * s - np.dot(u, u)

    resVec = 2.0 * np.multiply(tmp1, u)  + \
        np.multiply(tmp2, srcVec) + \
        2.0 * s * np.cross(u, srcVec)
    return resVec

#### print euler
q = np.quaternion(0.5, 0.5, 0.5, 0.5)
print(computeEulerFromQuaternion(q.w, q.x, q.y, q.z))

#### print euler
q = np.quaternion(0, math.pi/2, 0.0, 0.0)
print(computeEulerFromQuaternion(q.w, q.x, q.y, q.z))

#### rotate a vector
q = np.quaternion(0.5, 0.5, 0.5, 0.5) # (pi/2, 0, pi/2)
srcVec = [0, 0, 1]
resVec = rotateVecByQuaternion(q,  srcVec)
print(resVec)

#### quat multiplication
q1 = np.quaternion(0.5, 0.5, 0.5, 0.5) # (pi/2, 0, pi/2)
q2 = np.quaternion(0.5, 0.5, 0.5, 0.5) # (pi/2, 0, pi/2)
q = q1*q2
print(computeEulerFromQuaternion(q.w, q.x, q.y, q.z))
