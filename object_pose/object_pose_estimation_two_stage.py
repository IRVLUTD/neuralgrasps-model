#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np
import scipy.io
from scipy.spatial import KDTree
import open3d as o3d
from sklearn.cluster import KMeans
from pytorch3d.transforms.so3 import (
    so3_exp_map, so3_log_map,
)
from pytorch3d.transforms import euler_angles_to_matrix

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
add_path(lib_path)

import utils.misc as ws
import utils.data_utils
import utils.train_utils
import utils.eval_utils
import utils.mesh
import utils.dataset as d
import models.networks as arch
from demo_utils import get_graspit_grasps

DATA_SOURCE = os.path.join(lib_path, '../dataset_train/')

def find_cluster_labels(X, K=2):
    kmeans = KMeans(n_clusters=K, random_state=42)
    print(f"Running KMeans with K={K}")
    labels = kmeans.fit_predict(X)
    return labels

def extract_idx_from_Z(experiment_dir, gripper: str, gnum: int):       
    split_filename = os.path.join(experiment_dir, 'split_train.json')
    split = json.load(open(split_filename))
    _, grp_names, _, _ = utils.data_utils.get_instance_filelist(DATA_SOURCE, split)
    if gripper not in grp_names:
        print("Gripper not found!!!!")
        return
    start_idx = grp_names.index(gripper)
    return start_idx + gnum

def load_sdf_network(experiment_dir, object_name):
    '''
    experiment_dir: specific to the ycb object
    '''
    CHECKPOINT = 'latest'
    LATENT_CODE_DIR = ws.latent_codes_subdir
    specs_filename = os.path.join(experiment_dir, "specs.json")
    split_filename = os.path.join(experiment_dir, 'split_train.json')
    
    split = json.load(open(split_filename))
    specs = json.load(open(specs_filename))
    latent_size = specs["CodeLength"]
    gripper_weight = specs["GripperWeight"]

    # load decoder
    decoder = arch.dsdfDecoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(
            experiment_dir, ws.model_params_subdir, CHECKPOINT + ".pth")
    )

    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()
    
    # Load the latent vectors learned during the training 
    latent_vecs = ws.load_latent_vectors(experiment_dir, CHECKPOINT)
    print(latent_vecs.shape)
    
    # object info
    object_model = '%s_google_16k_textured_scale_1000' % object_name
    path_to_norm_npz = os.path.join(DATA_SOURCE, object_model, "norm_params.npz")
    data = np.load(path_to_norm_npz)
    obj_scale = data['scale']
    obj_offset = data['offset']
    print('obj_scale', obj_scale)
    print('obj_offset', obj_offset)
    
    # get graspit grasps
    train_split = 'split_train.json'
    graspit_grasps = get_graspit_grasps(experiment_dir, DATA_SOURCE, object_model, train_split)
    return decoder, latent_vecs, obj_scale, obj_offset, graspit_grasps
    
    
def load_objpc_contacts(ycb_model, gripper, grasp_idx, threshold=0.8):    
    # load the ycb model point cloud
    objpc_file = os.path.join(DATA_SOURCE, ycb_model, "object_point_cloud.ply")
    _obj_pcd = o3d.io.read_point_cloud(objpc_file)
    obj_pc = np.asarray(_obj_pcd.points)
    # load the contact map
    cmap_file = os.path.join(DATA_SOURCE, ycb_model, "contactmap", gripper, 
                                f"contactmap_graspnum_{grasp_idx}.npz")
    print('contact file', cmap_file)
    cmap_data = np.load(cmap_file)['heatmap']
    contact_pts = obj_pc[cmap_data > threshold]
    return contact_pts
    
    
def eval_query_pc_with_grad(device, decoder, latent_vec, queries, max_batch_size=30000):
    num_samples = queries.shape[0]
    xyz_samples = queries
    pred_sdf_obj = []
    pred_sdf_grp = []

    head = 0
    while head < num_samples:
        start_idx = head
        end_idx = min(head + max_batch_size, num_samples)
        sample_subset = xyz_samples[start_idx: end_idx, :3]
        sdf_obj, sdf_grp = utils.train_utils.decode_sdf(
            decoder, latent_vec, sample_subset)
        pred_sdf_obj.append(sdf_obj.squeeze(1))
        pred_sdf_grp.append(sdf_grp.squeeze(1))
        head += max_batch_size

    return xyz_samples, torch.cat(pred_sdf_obj), torch.cat(pred_sdf_grp)
    
    
# choose one pose for optimization
def contruct_rotation_matrices():

    degrees = np.arange(0, 360, 45)
    l = len(degrees)
    num = l * l * l
    eulers = np.zeros((num, 3), dtype=np.float32)
    count = 0
    for x in degrees:
        for y in degrees:
            for z in degrees:
                eulers[count, :] = np.array([x, y, z]) * np.pi / 180.0
                count += 1

    eulers_tensor = torch.from_numpy(eulers)
    Rs_tensor = euler_angles_to_matrix(eulers_tensor, 'XYZ')
    return Rs_tensor


# pc: point cloud of the object in robot base
def estimate_object_pose(num_iters, pc, obj_scale, obj_offset, l_vec, decoder):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # initialize the absolute log-rotations/translations with random entries
    T_init = torch.randn(1, 3, dtype=torch.float32, device=device)
    
    # convert to tensors
    pc_tensor = torch.from_numpy(pc.T).cuda().float()
    obj_offset_tensor = torch.from_numpy(obj_offset).cuda().float()
    obj_offset_tensor = torch.reshape(obj_offset_tensor, (3, 1))
    obj_scale_tensor = torch.from_numpy(obj_scale).cuda().float()
    
    # initialize the translation
    # compute the mean of the point cloud
    pc_mean = np.mean(pc, axis=0)
    z_min = np.min(pc[:, 2])
    T_init[0, 0] = -pc_mean[0]
    T_init[0, 1] = -pc_mean[1]
    # T_init[0, 2] = -z_min
    T_init[0, 2] = -pc_mean[2]
    print('init translation', T_init)
    
    # select an initial pose
    print('initializing rotations')
    Rs_tensor = contruct_rotation_matrices()
    num = Rs_tensor.shape[0]
    losses = np.zeros((num, ), dtype=np.float32)
    for i in range(num):
        R = Rs_tensor[i].cuda()
        # transform points
        pc_obj = torch.matmul(R, pc_tensor) + T_init.t()
        
        # normalize the points
        pc_obj = pc_obj / obj_scale_tensor
        
        # compute SDFs
        _, sdf_obj, _ = eval_query_pc_with_grad(device, decoder, l_vec.float().cuda(), pc_obj.t(), max_batch_size=40000)
        
        # compute loss
        losses[i] = torch.sum(torch.absolute(sdf_obj)).detach().cpu().numpy()
    index = np.argsort(losses)
    # use top K
    index = index[:5]
    index = np.random.choice(index)
    print('select rotation %d' % index)
    print(Rs_tensor[index])
    
    # initialize rotation
    log_R_init = torch.randn(1, 3, dtype=torch.float32, device=device)    
    R = torch.reshape(Rs_tensor[index], (1, 3, 3))
    logR = so3_log_map(R)
    log_R_init[0, 0] = logR[0, 0]
    log_R_init[0, 1] = logR[0, 1]
    log_R_init[0, 2] = logR[0, 2]     
    print('init log R', log_R_init)

    # instantiate a copy of the initialization of log_R / T
    log_R = log_R_init.clone().detach()
    log_R.requires_grad = True
    T = T_init.clone().detach()
    T.requires_grad = True

    # init the optimizer
    # optimizer = torch.optim.SGD([log_R, T], lr=0.0001, momentum=0.9)
    optimizer = torch.optim.Adam([log_R, T], lr=0.01)

    # run the optimization
    factor = 1000.0
    for it in range(num_iters):
        # re-init the optimizer gradients
        optimizer.zero_grad()

        # compute the absolute camera rotations as 
        # an exponential map of the logarithms (=axis-angles)
        # of the absolute rotations
        R = so3_exp_map(log_R)
        
        # transform points
        pc_obj = torch.matmul(R[0], pc_tensor) + T.t()
        
        # normalize the points
        # pc_obj = (pc_obj - obj_offset_tensor) / obj_scale_tensor
        pc_obj = pc_obj / obj_scale_tensor
        
        # compute SDFs
        _, sdf_obj, _ = eval_query_pc_with_grad(device, decoder, l_vec.float().cuda(), pc_obj.t(), max_batch_size=40000)
        
        # compute loss
        loss = torch.sum(torch.absolute(sdf_obj))
        # loss = factor * torch.mean(torch.absolute(sdf_obj))
        loss.backward()
            
        # apply the gradients
        optimizer.step()
        print('%06d/%06d: loss=%06f' % (it, num_iters, loss))
        
    print('Optimization finished.')
    print('log_R:', log_R)
    print('T:', T)    

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = so3_exp_map(log_R).detach().cpu().numpy()
    pose[:3, 3] = T.detach().cpu().numpy()
    pose = np.linalg.inv(pose)
    print(pose)
    return pose, loss.detach().cpu().numpy()
    
    
    
# pc: point cloud of the object in robot base
def refine_object_pose(num_iters, pose, pc, obj_scale, obj_offset, l_vec, decoder, 
                            org_contact_pts, filename, log_data=False):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # initialize the absolute log-rotations/translations with random entries
    logR = so3_log_map(torch.from_numpy(pose[:3, :3].reshape(1, 3, 3)))
    log_R_init = torch.randn(1, 3, dtype=torch.float32, device=device)
    log_R_init[0, 0] = logR[0, 0]
    log_R_init[0, 1] = logR[0, 1]
    log_R_init[0, 2] = logR[0, 2]   
    
    # T_init = torch.randn(1, 3, dtype=torch.float32, device=device)
    T_init = torch.from_numpy(pose[:3, 3].reshape(1, 3)).cuda()
    print('init translation', T_init)
    print('init log R', log_R_init)    
    
    # convert to tensors
    pc_tensor = torch.from_numpy(pc.T).cuda().float()
    obj_offset_tensor = torch.from_numpy(obj_offset).cuda().float()
    obj_offset_tensor = torch.reshape(obj_offset_tensor, (3, 1))
    obj_scale_tensor = torch.from_numpy(obj_scale).cuda().float()

    # instantiate a copy of the initialization of log_R / T
    log_R = log_R_init.clone().detach()
    log_R.requires_grad = True
    T = T_init.clone().detach()
    T.requires_grad = True

    # init the optimizer
    optimizer = torch.optim.Adam([log_R, T], lr=0.01)

    R = so3_exp_map(log_R)                
    # transform points
    pc_obj = torch.matmul(R[0], pc_tensor) + T.t()
    # normalize the points
    # pc_obj = (pc_obj - obj_offset_tensor) / obj_scale_tensor
    pc_obj = pc_obj / obj_scale_tensor
    # construct a kdtree over the input point cloud
    # idxs represent the points on the input point cloud that are closest to the "true"
    # contact pts        
    tree_pts = pc_obj.detach().cpu().numpy()
    tree_pc = KDTree(tree_pts.T)
    _, idxs = tree_pc.query(org_contact_pts)
    # Note pc_tensor and idxs (closest pts) still remain the same for the inner loop
    # Only the R,T change via the optimizer step
    for it in range(num_iters):
        optimizer.zero_grad()
        R = so3_exp_map(log_R)                
        # transform points
        pc_obj = torch.matmul(R[0], pc_tensor) + T.t()
        # normalize the points
        # pc_obj = (pc_obj - obj_offset_tensor) / obj_scale_tensor
        pc_obj = pc_obj / obj_scale_tensor    
        # compute SDFs
        _, sdf_obj, _ = eval_query_pc_with_grad(device, decoder, l_vec.float().cuda(), pc_obj.t(), max_batch_size=40000)
        _, _, sdf_h_closest = eval_query_pc_with_grad(device, decoder, l_vec.float().cuda(), pc_obj.t()[idxs], max_batch_size=40000)
        # compute loss -- Take the mean instead of sum
        loss_obj = torch.sum(torch.absolute(sdf_obj))
        loss_contact = torch.sum(torch.absolute(sdf_h_closest))
        loss = loss_obj + loss_contact
        loss.backward()            
        # apply the gradients
        optimizer.step()
        print('%06d/%06d: loss=%06f, loss_obj=%06f, loss_contact=%06f' % (it, num_iters, loss, loss_obj, loss_contact))
            
        if log_data:
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = so3_exp_map(log_R).detach().cpu().numpy()
            pose[:3, 3] = T.detach().cpu().numpy()
            # pose = np.linalg.inv(pose)
            saved_pose_file = f"./logs/dump_result_{filename.split('.')[0]}_{it}.npz"
            np.savez(saved_pose_file, inv_pose=pose, idxs=idxs)

    print('%06d/%06d: loss=%06f' % (it, num_iters, loss))
    print('log_R:', log_R)
    print('T:', T)
    print('Optimization finished.')

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = so3_exp_map(log_R).detach().cpu().numpy()
    pose[:3, 3] = T.detach().cpu().numpy()
    pose = np.linalg.inv(pose)
    print(pose)
    return pose, loss.detach().cpu().numpy()    


if __name__ == "__main__":
    """
    Main function to run the code
    """
    # load org pc and cmaps
    # ycb_model = "003_cracker_box_google_16k_textured_scale_1000"
    # experiment_dir = os.path.join(this_dir, '../experiments/all5_003_dsdf_50_varcmap/')
    object_name = '006_mustard_bottle' 
    ycb_model = '%s_google_16k_textured_scale_1000' % object_name
    experiment_dir = os.path.join(this_dir, '../experiments/all5_%s_dsdf_50_varcmap/' % object_name[:3])
    
    # load sdf decoder
    print('loading network')
    decoder, latent_vecs, obj_scale, obj_offset, graspit_grasps = load_sdf_network(experiment_dir, object_name)
    
    # load point cloud
    # filename = 'data.mat'
    # filename = 'cracker_box_45degrees.mat'
    # filename = 'cracker_box_horizontal.mat'
    filename = 'cracker_box_vertical.mat'
    # filename = "mustard_bottle_horizontal.mat"

    gripper_name = "fetch_gripper"
    grasp_num = 9
    threshold = 0.6
    l_vec = latent_vecs[grasp_num]
    l_vec = l_vec.squeeze()    
    gt_contacts = load_objpc_contacts(ycb_model, gripper_name, grasp_num, threshold)
    Z_idx = extract_idx_from_Z(experiment_dir, gripper_name, grasp_num)
    print("training index:", Z_idx)

    data = scipy.io.loadmat(os.path.join("data", filename))
    pc = data['pc']
    num_iters = 1000
    pose, loss = estimate_object_pose(num_iters, pc, obj_scale, obj_offset, l_vec, decoder)
    
    # saved_pose_file = f'result_{filename.split(".")[0]}_icp.npy'
    # np.save(os.path.join("logs", saved_pose_file), pose)
