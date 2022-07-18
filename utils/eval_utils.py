from base64 import decode
from scipy.spatial import KDTree
import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch
import os

import utils.train_utils


def extract_graspidx_from_npzfile(npzpath: str) -> int:
    '''    
    Splits the npzfile path and extracts out the grasp number from it.
    Can be made into a oneliner as follows:
    `return int(os.path.split(npzpath)[-1].split('.')[0].split('_')[-1])

    npzfile: Full path, example: '~/data/output_data/003_cracker/sdf/allegro/sdf_graspnum_0.npz'

    Returns:
        grasp_num (int): integer corresponding to the grasp number from the refined grasp
    '''
    fname = os.path.split(npzpath)[-1]
    fname_wo_ext = fname.split('.')[0]
    graspnum_str = fname_wo_ext.split('_')[-1]
    return int(graspnum_str)


def extract_pcfile_from_npzfile(npzpath: str, pc_type='single') -> int:
    '''    
    Splits the npzfile path and extracts out point cloud file path from it.
    
    Input:
        npzfile: Full path, example: '~/data/output_data/003_cracker/sdf/allegro/sdf_graspnum_0.npz'

        pc_type: 'single' or 'combined' -- single represents just the point cloud of gripper
              while combined refers to the point cloud of both gripper and object
     
    Returns:
        pc_file (str): Path to the point cloud
    '''    
    graspidx = extract_graspidx_from_npzfile(npzpath)
    if pc_type == 'combined':
        pc_fname = f'combined_graspnum_{graspidx}.ply'
    else: # default to the just the gripper pc
        pc_fname = f'single_graspnum_{graspidx}.ply'

    path_without_npz = os.path.split(npzpath)[0]
    pc_folder = path_without_npz.replace('sdf', 'point_cloud')
    pc_file = os.path.join(pc_folder, pc_fname)
    return pc_file


def compute_pc_chamfer(gt_points, gen_points, kdtree_gt_pts=None):
    '''
    This functions computes the two-way chamfer distances between two input point clouds.

    Input:
        gt_points: numpy array (N,3) containing the points of the ground truth shape

        gen_points: numpy array (N,3) containing the points inferred for the shape using the 
                    latent code over random query points inside the unit sphere.
        
        kdtree_gt_pts: KDTree object over the gt_points (Optional). Since the gt points might
                   not change across different grasp (scenes), we can pass it along to save
                   time
    Returns:
        (chamfer_to_gt, chamfer_to_gen) Tuple which are the directional chamfer distances

    '''
    if not kdtree_gt_pts:
        kdtree_gt_pts = KDTree(gt_points)
    
    distances_to_gt, _ = kdtree_gt_pts.query(gen_points)
    chamfer_to_gt = np.mean(np.square(distances_to_gt))

    kdtree_gen_pts = KDTree(gen_points)
    distances_to_gen, _ = kdtree_gen_pts.query(gt_points)
    chamfer_to_gen = np.mean(np.square(distances_to_gen))

    return chamfer_to_gt, chamfer_to_gen


########################################
# Custom: Eval the Model on Query points
########################################


def sample_uniform_points_in_unit_sphere(amount):
    # unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = torch.FloatTensor(amount * 2 + 20, 3).uniform_(-1, 1)
    unit_sphere_points = unit_sphere_points[torch.linalg.norm(
        unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = torch.FloatTensor(amount, 3)
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(
            amount - points_available)
        # return result
    else:
        result = unit_sphere_points[:amount, :]
    # Possible alternative:
    # result = torch.FloatTensor(amount * 2 + 20,3).uniform_(-1, 1)
    # return torch.from_numpy(result)
    return result


def eval_random_query_pc(decoder, latent_vec, num_samples=100000, max_batch_size=30000):
    decoder.eval()

    xyz_samples = sample_uniform_points_in_unit_sphere(num_samples)
    pred_sdf_obj = torch.zeros(xyz_samples.shape[0])
    pred_sdf_grp = torch.zeros(xyz_samples.shape[0])

    xyz_samples.requires_grad = False
    pred_sdf_grp.requires_grad = False
    pred_sdf_obj.requires_grad = False

    head = 0
    while head < num_samples:
        start_idx = head
        end_idx = min(head + max_batch_size, num_samples)
        sample_subset = xyz_samples[start_idx: end_idx].cuda()
        sdf_obj, sdf_grp = utils.train_utils.decode_sdf(
            decoder, latent_vec, sample_subset)
        pred_sdf_obj[start_idx: end_idx] = sdf_obj.squeeze(1).detach().cpu()
        pred_sdf_grp[start_idx: end_idx] = sdf_grp.squeeze(1).detach().cpu()
        head += max_batch_size

    return xyz_samples, pred_sdf_obj, pred_sdf_grp


def eval_query_pc(decoder, latent_vec, queries, max_batch_size=30000):
    decoder.eval()
    num_samples = queries.shape[0]
    xyz_samples = queries
    pred_sdf_obj = torch.zeros(xyz_samples.shape[0])
    pred_sdf_grp = torch.zeros(xyz_samples.shape[0])

    xyz_samples.requires_grad = False
    pred_sdf_grp.requires_grad = False
    pred_sdf_obj.requires_grad = False

    head = 0
    while head < num_samples:
        start_idx = head
        end_idx = min(head + max_batch_size, num_samples)
        sample_subset = xyz_samples[start_idx: end_idx, :3].cuda()
        sdf_obj, sdf_grp = utils.train_utils.decode_sdf(
            decoder, latent_vec, sample_subset)
        pred_sdf_obj[start_idx: end_idx] = sdf_obj.squeeze(1).detach().cpu()
        pred_sdf_grp[start_idx: end_idx] = sdf_grp.squeeze(1).detach().cpu()
        head += max_batch_size

    return xyz_samples, pred_sdf_obj, pred_sdf_grp


def eval_query_pc_multi_gripper(encoderDecoder, latent_vec, queries, gripper_pc, max_batch_size=30000):
    encoderDecoder.eval()
    num_samples = queries.shape[0]
    xyz_samples = queries
    pred_sdf_obj = torch.zeros(xyz_samples.shape[0])
    pred_sdf_grp = torch.zeros(xyz_samples.shape[0])

    xyz_samples.requires_grad = False
    pred_sdf_grp.requires_grad = False
    pred_sdf_obj.requires_grad = False

    head = 0
    while head < num_samples:
        start_idx = head
        end_idx = min(head + max_batch_size, num_samples)
        sample_subset = xyz_samples[start_idx: end_idx, :3].cuda()
        sdf_obj, sdf_grp = utils.train_utils.decode_sdf_encoderDecoder(
            encoderDecoder, latent_vec, sample_subset, gripper_pc)
        pred_sdf_obj[start_idx: end_idx] = sdf_obj.squeeze(1).detach().cpu()
        pred_sdf_grp[start_idx: end_idx] = sdf_grp.squeeze(1).detach().cpu()
        head += max_batch_size

    return xyz_samples, pred_sdf_obj, pred_sdf_grp

def eval_query_pc_multisdf(decoder, latent_vec, queries, gripper_idx, max_batch_size=30000):
    decoder.eval()
    num_samples = queries.shape[0]
    xyz_samples = queries
    pred_sdf_obj = torch.zeros(xyz_samples.shape[0])
    pred_sdf_grp = torch.zeros(xyz_samples.shape[0])

    xyz_samples.requires_grad = False
    pred_sdf_grp.requires_grad = False
    pred_sdf_obj.requires_grad = False

    head = 0
    while head < num_samples:
        start_idx = head
        end_idx = min(head + max_batch_size, num_samples)
        sample_subset = xyz_samples[start_idx: end_idx, :3].cuda()
        sdf_obj, sdf_grp = utils.train_utils.decode_multisdf(
            decoder, latent_vec, sample_subset, gripper_idx)
        pred_sdf_obj[start_idx: end_idx] = sdf_obj.squeeze(1).detach().cpu()
        pred_sdf_grp[start_idx: end_idx] = sdf_grp.squeeze(1).detach().cpu()
        head += max_batch_size

    return xyz_samples, pred_sdf_obj, pred_sdf_grp
