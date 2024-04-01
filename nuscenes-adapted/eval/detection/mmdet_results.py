import sys
import os

from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import DetectionEval

plot_examples = 0 # nuscenes data required
render_curves = bool(1)
verbose       = bool(1)

def  metaDataCVvalidation(dataroot: str):
    """
    Init meta data for cross fold validation for 3D mmdet results
    """

    eval_list = []
    # Add PointPillar results to list
    result_path = dataroot + '\\PointPillar\\'
    detector_name = 'PointPillar'

    eval_config = {}
    eval_config['result_path']  = result_path + 'PointPillar\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'PointPillar\\nuscenes-metrics'
    eval_config['eval_set']     = 'val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_1\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_1\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_1_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_2\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_2\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_2_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_3\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_3\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_3_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_4\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_4\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_4_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_5\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_5\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_5_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    # Add CenterPoint Voxel 01 results to list
    result_path = dataroot + '\\CenterPoint\\Voxel01\\'
    detector_name = 'CenterPoint Voxel 01'

    eval_config = {}
    eval_config['result_path']  = result_path + 'Voxel01\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'Voxel01\\nuscenes-metrics'
    eval_config['eval_set']     = 'val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_1\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_1\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_1_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_2\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_2\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_2_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_3\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_3\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_3_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_4\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_4\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_4_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_5\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_5\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_5_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    # Add CenterPoint Voxel 0075 results to list
    result_path = dataroot + '\\CenterPoint\\Voxel0075\\'
    detector_name = 'CenterPoint Voxel 0075'

    eval_config = {}
    eval_config['result_path']  = result_path + 'Voxel0075\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'Voxel0075\\nuscenes-metrics'
    eval_config['eval_set']     = 'val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_1\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_1\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_1_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_2\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_2\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_2_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_3\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_3\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_3_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_4\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_4\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_4_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_5\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_5\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_5_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    # Add CenterPoint Pillar 02 results to list
    result_path = dataroot + '\\CenterPoint\\Pillar02\\'
    detector_name = 'CenterPoint Pillar 02'

    eval_config = {}
    eval_config['result_path']  = result_path + 'Pillar02\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'Pillar02\\nuscenes-metrics'
    eval_config['eval_set']     = 'val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_1\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_1\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_1_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_2\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_2\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_2_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_3\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_3\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_3_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_4\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_4\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_4_val' 
    eval_config['detector_name']     = detector_name
    eval_list.append(eval_config)

    eval_config = {}
    eval_config['result_path']  = result_path + 'LOOCV_5\\results_nusc.json'
    eval_config['output_dir']   = result_path + 'LOOCV_5\\nuscenes-metrics'
    eval_config['eval_set']     = 'LOOCV_5_val' 
    eval_config['detector_name']     = detector_name 
    eval_list.append(eval_config)

    return eval_list