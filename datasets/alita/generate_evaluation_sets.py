# Generate evaluation sets
# This script is adapted from: https://github.com/jac99/Egonn/blob/main/datasets/southbay/generate_evaluation_sets.py

import argparse
import numpy as np
from typing import List
import os
import sys
import glob
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from datasets.base_datasets import EvaluationTuple, EvaluationSet, filter_query_elements

def get_pose_transform(pose6d):
    rot_matrix = R.from_euler('xyz', pose6d[3:]).as_matrix()
    trans_vector = pose6d[:3].reshape((3, 1))

    trans_matrix = np.identity(4)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[:3, 3:] = trans_vector

    return trans_matrix

def get_length(poses):
    current_pos = poses[0][:2,3]
    total = 0
    for i in range(len(poses)-1):
        delta = np.linalg.norm(poses[i][:2,3] - poses[i+1][:2,3])
        print(delta)
        total += delta
    print('')

def get_poses_ugv(pose_files):
    poses = []
    timestamps = []
    for f in pose_files:
        pose = np.load(f)
        timestamps.append(pose[6])
        T = get_pose_transform(pose[:6])
        poses.append(T)
    return np.asarray(timestamps), np.asarray(poses)

def get_scans(base_dir, area, split, min_displacement: float = 0.0) -> List[EvaluationTuple]:

    operating_dir = os.path.join(base_dir, split, area)
    pcd_files = sorted(glob.glob(os.path.join(operating_dir, '*.pcd')))
    pose_files = sorted(glob.glob(os.path.join(operating_dir, '*.npy')))
    timestamps, poses = get_poses_ugv(pose_files)
    get_length(poses)


    elems = []
    for ndx in range(len(pcd_files)):
        pose = poses[ndx]
        position = pose[0:2, 3]       # (x, y) position in global coordinate frame
        rel_scan_filepath = pcd_files[ndx][len(base_dir):]
        timestamp = timestamps[ndx]

        item = EvaluationTuple(timestamp, rel_scan_filepath, position=position, pose=pose)
        elems.append(item)

    print(f"{len(elems)} total elements in {split} split")

    # Filter-out elements leaving only 1 per grid cell with min_displacement size
    pos = np.zeros((len(elems), 2), dtype=np.float32)
    for ndx, e in enumerate(elems):
        pos[ndx] = e.position

    # Quantize x-y coordinates. Quantized coords start from 0
    pos = np.floor(pos / min_displacement)
    pos = pos.astype(int)
    _, unique_ndx = np.unique(pos, axis=0, return_index=True)

    # Leave only unique elements
    elems = [elems[i] for i in unique_ndx]
    print(f"{len(elems)} filtered elements in {split} split with grid cell size = {min_displacement}")

    return elems


def generate_evaluation_set(dataset_root: str, area: str, min_displacement: float = 0.0, dist_threshold=5) -> \
        EvaluationSet:
    map_set = get_scans(dataset_root, area, 'DATABASE', min_displacement)
    query_set = get_scans(dataset_root, area, 'QUERY', min_displacement)
    query_set = filter_query_elements(query_set, map_set, dist_threshold)
    print(f'Area: {area} - {len(map_set)} database elements, {len(query_set)} query elements\n')
    return EvaluationSet(query_set, map_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation sets for UGV dataset')
    parser.add_argument('--dataset_root', type=str, required=False, default='')
    parser.add_argument('--min_displacement', type=float, default=0.01)
    # Ignore query elements that do not have a corresponding map element within the given threshold (in meters)
    parser.add_argument('--dist_threshold', type=float, default=5)

    args = parser.parse_args()
    print(f'Dataset root: {args.dataset_root}')
    print(f'Minimum displacement between scans in each set (map/query): {args.min_displacement}')
    print(f'Ignore query elements without a corresponding map element within a threshold [m]: {args.dist_threshold}')

    area = 'val_5'   # Evaluation area
    eval_set = generate_evaluation_set(dataset_root=args.dataset_root,area=area, min_displacement=args.min_displacement,
                                       dist_threshold=args.dist_threshold)
    pickle_name = f'test_{area}_{args.min_displacement}_{args.dist_threshold}.pickle'
    # file_path_name = os.path.join(args.dataset_root, pickle_name)
    file_path_name = os.path.join(os.path.dirname(__file__), pickle_name)
    print(f"Saving evaluation pickle: {file_path_name}")
    eval_set.save(file_path_name)
