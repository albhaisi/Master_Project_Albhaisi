import numpy as np

from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.data_classes import DetectionBox


def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))

sa = DetectionBox(translation=(352.627,1067.133,0.804))
sr = DetectionBox(translation=(352.690583364412, 1067.312016267324, 0.9425697179963862))
res = center_distance(sa, sr)
print(res) 