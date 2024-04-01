'''
Possibility to input lidar & radar points into 3D Box - later 2D
function in nuscenes-devkit: extracting info about lidar and radar
	input: sample token
	output: number of radar / lidar points

current state - 20.06.2022:
    only take tokens from sample results file
    proof that num_lidar_pts in annotation fits with calculated one

to do:
    accommodate autobin sample dataset:
    - using this script, we can extract the num_lidar_pts and add it into the sample_annotation.json file directly
    - ask user to input which dataset they want to edit (changes needed for multiple datasets from autobin)

'''

from nuscenes.nuscenes import NuScenes
import json
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

from pyquaternion import Quaternion

import matplotlib as plt

import numpy as np

import os

import pdb


def points_in_box(box: 'Box', points: np.ndarray, wlh_factor: float = 1.0):
    """
    FROM: https://forum.nuscenes.org/t/information-on-reference-frames/361/3
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
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

    return mask

def extract_num_pts(nusc, sample, counter):

    # load lidar
    lidar_token = sample['data']['LIDAR_TOP']
    
    # get corners from annotation
    # translation and size is in global coordinates
    # units: meter, global coordinate system: [0 0 0] "top left" corner of map
    lidar_rec = nusc.get('sample_data',lidar_token)

    ann_tokens = sample['anns']
    num_lidar_pts_list = []
    count = 0
    for ann_token in ann_tokens:
        data_path,boxes,camera_intrinsic = nusc.get_sample_data(lidar_token,selected_anntokens=[ann_token])
        pointcloud = LidarPointCloud.from_file(data_path)
        mask = points_in_box(boxes[0],pointcloud.points[:3])
        # number of lidar points inside bbox
        num_lidar_pts = np.where(mask)[0].shape[0]
        num_lidar_pts_list.append(num_lidar_pts)

        # load annotation
        ann = nusc.get('sample_annotation',ann_token)
        ann_points = pointcloud.points[:3,mask]
        
        if num_lidar_pts == ann['num_lidar_pts']:
            count += 1
            #print('annotated number of lidar points: ', ann['num_lidar_pts'],'\n')
            #print('calculated number of lidar points: ', num_lidar_pts,'\n')
        else:
            print('wrong - ann_token: ', ann_token, 'annotated: ',ann['num_lidar_pts'],'calculated: ', num_lidar_pts)
    
    if count == len(ann_tokens):
        print('perfect num_lidar_pts extraction for all annotations in sample: ', sample['token'])
        counter += 1

    return num_lidar_pts_list, counter

if __name__ == "__main__":

    dataroot        = "/mmdetection3d/data/nuscenes"
    version         = "v1.0-mini"
    nusc            = NuScenes(dataroot=dataroot, version=version) 

    with open('/mmdetection3d/data/nuscenes/results_nusc.json') as f:
        results     = json.load(f)
    
    # list all sample_token in result file
    token_list = list(results['results'].keys())

    # counter
    counter = 0
    included_tokens = []
    # get a list of tokens from results that corresponds to the dataset (in this case mini dataset)
    for token in token_list:
        try:
            sample = nusc.get('sample', token)

            num_lidar_pts_list, counter = extract_num_pts(nusc, sample, counter)
            included_tokens.append(token)
        except:
            continue
    
    if counter == len(included_tokens):
        print('perfect num_lidar_pts extraction for all samples and annotations!!!')


        
