from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.data_classes import Box as NuScenesBox

import numpy as np
from pyquaternion import Quaternion

from typing import Callable
import json
import os.path as osp


# # Check scene description for relevant key words
# description = []
# for scene in nusc.scene:
#     description = description + scene['description'].lower().split(', ')

# for item in description:
#     item = item.rstrip()

# description = set(description)

# matches = ["ped", "car", "truck", "bus","bicycles","motorcycle","vehicles","crosswalk","trash","worker","turning","van","slope","intersection","sidewalk","garbage"]

# ignore = [item for item in description if any(x in item for x in matches)]
# description = [item for item in description if not any(x in item for x in matches)]

def group_scenes(nusc: NuScenes,
                eval_set: str,
                verbose: bool = False) -> dict:
    """
    Evaluates prediction at one time step
    :return: A tuple of high-level and the raw metric data.
    """
    splits = create_splits_scenes()
    if eval_set == 'all':
        scenes_all = [scene for scene in nusc.scene]
    else:
        scenes_all = [scene for scene in nusc.scene if scene['name'] in splits[eval_set]]
    

    # divide according to the situation

    scenes = defaultdict(list)
    scenes['rain']  = [scene['name'] for scene in scenes_all if 'rain' in scene['description'].lower()]
    scenes['night'] = [scene['name'] for scene in scenes_all if 'night' in scene['description'].lower()]
    scenes['rain_and_night'] = [scene['name'] for scene in scenes_all if scene['name'] in scenes['rain'] and scene['name'] in scenes['night']]
    scenes['default'] = [scene['name'] for scene in scenes_all if scene['name'] not in scenes['rain'] and scene['name'] not in scenes['night']]

    # remove redundant / mixed scenes
    scenes['rain']  = [scene for scene in scenes['rain']  if scene not in scenes['rain_and_night']]
    scenes['night'] = [scene for scene in scenes['night'] if scene not in scenes['rain_and_night']]

    if verbose:
        print('{} has {} scenes'.format(eval_set,len(scenes_all)))

        print('Rain has {} scenes'.format(len(scenes['rain'])))
        print('night has {} scenes'.format(len(scenes['night'])))
        print('rain_and_night has {} scenes'.format(len(scenes['rain_and_night'])))
        print('default has {} scenes'.format(len(scenes['default'])))

    assert len(scenes['rain'])+len(scenes['night'])+len(scenes['rain_and_night'])+len(scenes['default']) == len(scenes_all)

    sample_tokens_all =  [s['token'] for s in nusc.sample]
    sample_tokens = defaultdict(list)
    for situation in scenes.keys():
        for sample_token in sample_tokens_all:
            scene_token = nusc.get('sample', sample_token)['scene_token']
            scene_record = nusc.get('scene', scene_token)
            if scene_record['name'] in  scenes[situation]:
                sample_tokens[situation].append(sample_token)

    return scenes,sample_tokens

def match_pred_boxes(sample_tokens: list,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> list:
    """
    Matches prediction at one time step
    :return: A tuple of high-level and the raw metric data.
    """
    detector_names = pred_boxes.keys()
    prediction_instances = {}
    for sample_token in sample_tokens:
        prediction_instance = []

        for detector in detector_names:
            taken = set()

            for box in pred_boxes[detector][sample_token]:
                if box.detection_name == class_name:
                
                    min_dist = np.inf
                        
                    for idx ,ref_box_list in enumerate(prediction_instance):
                            for ref_box in [ref_box_list[key] for key in ref_box_list.keys()]:
                                if ref_box:
                                    # Find closest match among ref boxes
                                    if idx not in taken:
                                        this_distance = dist_fcn(ref_box, box)
                                        if this_distance < min_dist:
                                            min_dist = this_distance
                                            match_idx = idx
                    
                    is_match = min_dist < dist_th

                    if is_match:
                        taken.add(match_idx)
                        prediction_instance[match_idx][detector] = box
                        #print('Match instance, dist: {}'.format(min_dist))
                        #print('Instance with match index {} has {} items'.format(match_idx,len(prediction_instance[match_idx])))
                        
                    else:
                        # Add instance if no match 
                        # prediction_instance[match_idx].append([])
                        item = {}
                        item[detector] = box
                        prediction_instance.append(item)
                        taken.add(len(prediction_instance)-1)
                        #print('Add new instance, min_dist: {}'.format(min_dist))
            
        prediction_instances[sample_token] = prediction_instance
    
    return prediction_instances

def get_num_pts_in_box(box: 'Box', 
                       points: np.ndarray, 
                       wlh_factor: float = 1.0):
    """
    FROM: https://forum.nuscenes.org/t/information-on-reference-frames/361/3
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return num_pts: <int>
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    num_pts = np.where(mask)[0].shape[0]
    return num_pts

def add_num_pts_to_file(nusc: NuScenes, 
                        filename: str):
    """ 
    get number of points inside bounding boxes from test result .json file 
    :param nusc: <NuScenes>.
    :param filename: <str>.
    :return results: <dict>.   
    """
    assert osp.isfile(filename), 'file does not exist'

    with open(filename) as f:
        results     = json.load(f)
    
    # list all sample_token in result file
    token_list = list(results['results'].keys())

    for i in range(0,len(results['results'])):
        for j in range(0,len(results['results'][token_list[i]])):
            anno = results['results'][token_list[i]][j]
            boxes = NuScenesBox( center=anno['translation'],
                                size = anno['size'],
                                orientation=Quaternion(anno['rotation']),
                                score=anno['detection_score'])
            
            sample_token = nusc.get('sample',anno['sample_token'])['data']['LIDAR_TOP']

            data_path = nusc.get_sample_data_path(sample_token)
            pointcloud = LidarPointCloud.from_file(data_path)
            num_pts = get_num_pts_in_box(boxes[0],pointcloud.points[:3])
            results['results'][token_list[i]][j]['num_lidar_pts'] = num_pts
    
    return results
            

def add_num_pts_to_inference(result,
                             data):
    """
    add number of points inside bounding boxes from inference result to the result
    :param result:
    :param data:
    :return result:
    """
    for j in range(len(result[0]['pts_bbox']['boxes_3d'])):
        if result[0]['pts_bbox']['scores_3d'][j] > 0.3:
            anno = result[0]['pts_bbox']['boxes_3d'][j]
            boxes = NuScenesBox(center = anno.center.numpy()[0],
                                size = anno.tensor[0][3:6].numpy(),
                                orientation = Quaternion(axis=[0, 0, 1], radians=anno.yaw))
            data_path = data['img_metas'][0][0]['pts_filename']
            pointcloud = LidarPointCloud.from_file(data_path)
            num_pts = get_num_pts_in_box(boxes[0],pointcloud.points[:3])

            result[0]['pts_bbox']['num_pts'][j] = num_pts

    return result



# if __name__ == "__main__":

#     from mmdet3d.apis import inference_detector, init_model
#     ''' function if data directly from inference detection '''
#     filenames = glob.glob('/mmdetection3d/data/nuscenes/samples/LIDAR_TOP/*.bin')
#     checkpoint_file = '/mmdetection3d/result/LOOCV_1/latest.pth'
#     config_file = '/mmdetection3d/result/LOOCV_1/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_LOOCV_1.py'

#     model = init_model(config_file,checkpoint_file, device='cuda:0')

#     result, data = inference_detector(model, filenames[1])

#     result = add_num_pts_to_inference(result,data)

#     ''' function to add the num_pts into the json results file'''
#     nusc = NuScenes(dataroot='/mmdetection3d/data/nuscenes', version='v1.0-trainval')
#     filename = '/mmdetection3d/data/nuscenes/results_nusc.json'
#     results = add_num_pts_to_file(nusc, filename)

    # included_tokens = []
    # # get a list of tokens from results that corresponds to the dataset (in this case mini dataset)
    # for token in token_list:
    #     try:
    #         sample = nusc.get('sample', token)
    #         included_tokens.append(token)
    #     except:
    #         continue


