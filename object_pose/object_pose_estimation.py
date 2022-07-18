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
    so3_exp_map,
)

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

def load_sdf_network(experiment_dir, idx: int):
    '''
    experiment_dir: specific to the ycb object
    idx: the index for the latent vector
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
    
    # For object point cloud sdf inference, any one latent vector is fine. 
    # Later we can store the best scoring vector as a part of the model state.
    # l_vec = latent_vecs[0]
    l_vec = latent_vecs[idx]
    l_vec = l_vec.squeeze()
    print(l_vec.shape)
    
    # object info
    object_model = '003_cracker_box_google_16k_textured_scale_1000'
    path_to_norm_npz = os.path.join(DATA_SOURCE, object_model, "norm_params.npz")
    data = np.load(path_to_norm_npz)
    obj_scale = data['scale']
    obj_offset = data['offset']
    print('obj_scale', obj_scale)
    print('obj_offset', obj_offset)
    
    # get graspit grasps
    train_split = 'split_train.json'
    graspit_grasps = get_graspit_grasps(experiment_dir, DATA_SOURCE, object_model, train_split)
    return decoder, l_vec, obj_scale, obj_offset, graspit_grasps
    
def load_objpc_contacts(ycb_model, gripper, grasp_idx, threshold=0.8, cluster_idx=0):    
    # load the ycb model point cloud
    objpc_file = os.path.join(DATA_SOURCE, ycb_model, "object_point_cloud.ply")
    _obj_pcd = o3d.io.read_point_cloud(objpc_file)
    obj_pc = np.asarray(_obj_pcd.points)
    # load the contact map
    cmap_file = os.path.join(DATA_SOURCE, ycb_model, "contactmap", gripper, 
                                f"contactmap_graspnum_{grasp_idx}.npz")
    cmap_data = np.load(cmap_file)['heatmap']
    contact_pts = obj_pc[cmap_data > threshold]
    print("Object contact points loaded. Now clustering them")
    # Assuming gripper = parallel jaw, K=2 for clustering
    cluster_labels = find_cluster_labels(contact_pts, K=2)
    # Choose pts belonging to the cluster_idx cluster label.
    return contact_pts[cluster_labels == cluster_idx]
    
def eval_query_pc_with_grad(device, decoder, latent_vec, queries, max_batch_size=30000):
    num_samples = queries.shape[0]
    xyz_samples = queries
    pred_sdf_obj = []
    pred_sdf_grp = []
    decoder.eval()
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
    

# pc: point cloud of the object in robot base
def estimate_object_pose(num_iters, icp_iters, pc, obj_scale, obj_offset, l_vec, decoder, 
                            org_contact_pts, filename, log_data=False):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # Query the network using the provided Z to obtain the obj pts and contact pts
    # with torch.no_grad():
    #     queries, sdf_o, sdf_g = utils.eval_utils.eval_random_query_pc(decoder, l_vec.cuda())
    # delta = 0.05
    # sdf_g = torch.clamp(sdf_g, -delta, delta)
    # sdf_o = torch.clamp(sdf_o, -delta, delta)
    # eps = 2e-2
    # mask_obj = sdf_o < eps
    # mask_grp = sdf_g < eps
    # query_contact_pts = queries[mask_obj & mask_grp]
    
    # Transform the contact pts to object coordinates
    org_contact_pts = (org_contact_pts - obj_offset) / obj_scale

    # initialize the absolute log-rotations/translations with random entries
    log_R_init = torch.randn(1, 3, dtype=torch.float32, device=device)
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
    print('init log R', log_R_init)

    # instantiate a copy of the initialization of log_R / T
    log_R = log_R_init.clone().detach()
    log_R.requires_grad = True
    T = T_init.clone().detach()
    T.requires_grad = True

    # init the optimizer
    # optimizer = torch.optim.SGD([log_R, T], lr=0.0001, momentum=0.9)
    optimizer = torch.optim.Adam([log_R, T], lr=0.01)
    # threshold_sdf = torch.nn.Threshold(1e-3, 0)
    for it in range(num_iters):
        # re-init the optimizer gradients
        optimizer.zero_grad() # Commented out since no gradient step is taken in the outer loop
        # Use the current R,T (from the last inner loop) to find the closest points
        # compute the absolute camera rotations as 
        # an exponential map of the logarithms (=axis-angles)
        # of the absolute rotations
        R = so3_exp_map(log_R)                
        # transform points
        pc_obj = torch.matmul(R[0], pc_tensor) + T.t()
        # normalize the points
        pc_obj = (pc_obj - obj_offset_tensor) / obj_scale_tensor
        # pc_obj /= obj_scale_tensor
        # construct a kdtree over the input point cloud
        # idxs represent the points on the input point cloud that are closest to the "true"
        # contact pts        
        tree_pts = pc_obj.detach().cpu().numpy()
        tree_pc = KDTree(tree_pts.T)
        _, idxs = tree_pc.query(org_contact_pts)
        # _, idxs = tree_pc.query(query_contact_pts)
        print("Computed and quried the KDTree")
        if log_data:
            pose = np.eye(4, dtype=np.float32)
            with torch.no_grad():
                pose[:3, :3] = so3_exp_map(log_R).detach().cpu().numpy()
                pose[:3, 3] = T.detach().cpu().numpy()
            # pose = np.linalg.inv(pose)
            saved_pose_file = f"./logs/dump_result_{filename.split('.')[0]}_{it}.npz"
            np.savez(saved_pose_file, inv_pose=pose, idxs=idxs)
        # Note pc_tensor and idxs (closest pts) still remain the same for the inner loop
        # Only the R,T change via the optimizer step
        icp_loss = 0
        # Adjust the weights to 2 losses
        # weight_contact = it/num_iters * 1e-6 + 1e-6
        # weight_contact = 0
        # weight_object  = 1 - weight_contact
        for j in range(icp_iters):
            optimizer.zero_grad()
            R = so3_exp_map(log_R)                
            # transform points
            pc_obj = torch.matmul(R[0], pc_tensor) + T.t()
            # normalize the points
            pc_obj = (pc_obj - obj_offset_tensor) / obj_scale_tensor
            # compute SDFs
            _, sdf_obj, _ = eval_query_pc_with_grad(device, decoder, l_vec.float().cuda(), pc_obj.t(), max_batch_size=40000)
            _, _, sdf_h_closest = eval_query_pc_with_grad(device, decoder, l_vec.float().cuda(), pc_obj.t()[idxs], max_batch_size=40000)
            # compute loss -- Take the mean instead of sum
            loss_obj = torch.sum(torch.absolute(sdf_obj))
            loss_contact = torch.sum(torch.absolute(sdf_h_closest))
            loss = loss_obj + loss_contact
            icp_loss = loss.item()
            loss.backward()            
            # apply the gradients
            optimizer.step()
            # print('%06d/%06d/%06d: loss=%06f' % (it, j, icp_iters, loss))
        print('%06d/%06d: loss=%06f' % (it, num_iters, icp_loss))
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
    ycb_model = "003_cracker_box_google_16k_textured_scale_1000"
    experiment_dir = os.path.join(this_dir, '../experiments/all5_003_dsdf_50_varcmap/')
    # ycb_model = "006_mustard_bottle_google_16k_textured_scale_1000"
    # experiment_dir = os.path.join(this_dir, '../experiments/all5_006_dsdf_50_varcmap/')
    # load point cloud
    # filename = 'data.mat'
    # filename = 'cracker_box_45degrees.mat'
    filename = 'cracker_box_horizontal.mat'
    # filename = 'cracker_box_vertical.mat'
    # filename = "mustard_bottle_horizontal.mat"

    gripper_name = "fetch_gripper"
    grasp_num = 0
    threshold = 0.6
    gt_contacts = load_objpc_contacts(ycb_model, gripper_name, grasp_num, threshold,
                    cluster_idx=0)
    Z_idx = extract_idx_from_Z(experiment_dir, gripper_name, grasp_num)
    print(f"file:{filename} | threshold: {threshold} | training index:", Z_idx)
    # load sdf decoder and latent vector corresponding the grasp
    decoder, l_vec, obj_scale, obj_offset, graspit_grasps = load_sdf_network(experiment_dir, Z_idx)

    data = scipy.io.loadmat(os.path.join("data", filename))
    pc = data['pc']
    # pc -= np.mean(pc, axis=0)
    # real_pc_labels = find_cluster_labels(pc, K=2)
    # pc = pc[real_pc_labels == 0]
    # pose estimation. Total iters = num_iters * icp_iters
    num_iters = 100
    icp_iters = 1000
    pose, loss = estimate_object_pose(num_iters, icp_iters, pc, obj_scale, obj_offset, 
                    l_vec, decoder, gt_contacts, filename, log_data=True)
    
    saved_pose_file = f'result_{filename.split(".")[0]}_icp.npy'
    np.save(os.path.join("logs", saved_pose_file), pose)
