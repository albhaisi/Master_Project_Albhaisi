import numpy as np

from pyquaternion import Quaternion
from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox



def quaternion_yaw(q: Quaternion) -> float:

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def yaw_angle(gt_box: EvalBox) -> float:
    
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    return yaw_gt

sa = DetectionBox(rotation=(0.7033531034233292,0,0,0.7108406374882992 ))
angle = yaw_angle(sa)
print('Angle in radians:', angle)




