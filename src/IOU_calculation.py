import numpy as np

from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox

def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union
    return iou

sa = DetectionBox(size=( 2.014, 4.717,2.007))
sr = DetectionBox(size=(1.9543049902212424 ,4.604727033708916 ,1.7127268641698556))
res = scale_iou(sa, sr)
print(res)
