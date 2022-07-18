import json
import os
import random
import argparse
from collections import defaultdict

import open3d as o3d
import torch
import numpy as np

import sys
import pickle

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

THIS_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(THIS_DIR, '..')
add_path(LIB_PATH)

import utils.eval_utils
import utils.data_utils
import utils.misc as ws
import models.networks as arch

reconstructions_subdir = ws.reconstructions_subdir
reconstruction_codes_subdir = ws.reconstruction_codes_subdir

def get_reconstructed_code_filename(experiment_dir, epoch, ycb_model, gripper_name, instance_name):
    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        f"{ycb_model}-sdf-{gripper_name}-{instance_name}.pth")
    # ycb_model + '-sdf-' + gripper_name + '-' + instance_name + ".pth")


def main_function(experiments_dir, ycb_model, checkpoint, validation=False):
    print("===============================================================")
    print(f"metric_chamfer-val_{validation}-{ycb_model}")

    LATENT_CODE_DIR = ws.latent_codes_subdir
    EPS = 1e-4  # threshold for sdf values to be within the surface.
    NUM_RANDOM_SAMPLES = 1000000

    specs_filename = os.path.join(experiments_dir, "specs.json")

    if validation:
        datadir = os.path.join(LIB_PATH, '../dataset_validation/')
        split_filename = os.path.join(experiments_dir, 'split_validation.json')
    else:
        datadir = os.path.join(LIB_PATH, '../dataset_train/')
        split_filename = os.path.join(experiments_dir, 'split_train.json')


    specs = json.load(open(specs_filename))
    latent_size = specs["CodeLength"]
    decoder = arch.dsdfDecoder(
        latent_size,
        **specs["NetworkSpecs"]
    ).cuda()

    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(
            experiments_dir, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()

    with open(split_filename, "r") as f:
        split = json.load(f)

    cmap_f, grp_names, gpc_f, npz_filenames = utils.data_utils.get_instance_filelist(
        datadir, split)

    # Load the object point cloud
    model_pc_file = os.path.join(
        datadir, ycb_model, 'object_point_cloud.ply')
    obj_pcd = o3d.io.read_point_cloud(model_pc_file)
    gt_obj_pc_points = np.asarray(obj_pcd.points)
    
    # Load the normalization params:
    norm_file = os.path.join(datadir, ycb_model, 'norm_params.npz')
    norm_npz = np.load(norm_file)
    offset = norm_npz['offset']
    scale = norm_npz['scale']

    # Normalize the object point cloud
    gt_obj_pc_points = (gt_obj_pc_points - offset) / scale
    print("Loaded Object Point Cloud with shape:", gt_obj_pc_points.shape)

    if not validation:
        latent_vecs = ws.load_latent_vectors(experiments_dir, checkpoint)
        print("Loaded Latent Vectors:", latent_vecs.shape)

    # Compute chamfer distances over all training instances
    # Store w.r.t each gripper in this dict as lists   
    chamfer_dists = defaultdict(list)
    
    for idx in range(len(npz_filenames)):
        gripper = grp_names[idx]
        npz = npz_filenames[idx]
        print(idx, npz[-35:])

        if validation:
            npz_instance_name = os.path.split(npz)[-1].split('.')[0]
            lvec_file = get_reconstructed_code_filename(
                experiments_dir, saved_model_epoch,
                ycb_model, gripper, npz_instance_name
                )
            lvec_idx = torch.load(lvec_file)
            lvec_idx = lvec_idx.squeeze()
        else:
            lvec_idx = latent_vecs[idx].squeeze()
        # print(lvec_idx.shape)

        # Load the point cloud corresponding to this grasp/gripper
        grp_pc_file = utils.eval_utils.extract_pcfile_from_npzfile(npz)
        grp_pcd = o3d.io.read_point_cloud(grp_pc_file)
        gt_grp_pc_points = np.asarray(grp_pcd.points)
        # Normalize the loaded points
        gt_grp_pc_points = (gt_grp_pc_points - offset) / scale
        # print(gt_grp_pc_points.shape)

        # Try with random queries:
        with torch.no_grad():
            queries, sdf_obj, sdf_grp = utils.eval_utils.eval_random_query_pc(
                decoder, lvec_idx.cuda(), num_samples=NUM_RANDOM_SAMPLES)

        gen_grp_points = queries[sdf_grp < EPS]
        # print(gen_grp_points.shape)
        gen_obj_points = queries[sdf_obj < EPS]
        # print(gen_obj_points.shape)

        cd_obj_to_gt, cd_obj_to_gen = utils.eval_utils.compute_pc_chamfer(
            gt_obj_pc_points, gen_obj_points)

        cd_grp_to_gt, cd_grp_to_gen = utils.eval_utils.compute_pc_chamfer(
            gt_grp_pc_points, gen_grp_points)
        
        chamfer_dists['ycb_object'].append(cd_obj_to_gt + cd_obj_to_gen)
        chamfer_dists[gripper].append(cd_grp_to_gt + cd_grp_to_gen)

        # chamfer_to_gt['ycb_object'].append(cd_obj_to_gt)
        # chamfer_to_gt[gripper].append(cd_grp_to_gt)

        # chamfer_to_gen['ycb_object'].append(cd_obj_to_gen)
        # chamfer_to_gen[gripper].append(cd_grp_to_gen)
    
    for k in chamfer_dists:
        cd_mean = np.mean(chamfer_dists[k])
        cd_median = np.median(chamfer_dists[k])
        print(f"Model: {k} | CD mean: {cd_mean:.5f} | CD Median: {cd_median:.5f}")
    
    # add description only after the np.mean code runs
    chamfer_dists['description'] = f"metric_chamfer-val_{validation}-{ycb_model}"
    result_file = f"metric_chamfer-val_{validation}-{ycb_model}.p"
    with open(os.path.join('./', 'results', result_file), "wb") as f:
        pickle.dump(chamfer_dists, f)
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Compute Metrics: Chamfer Distance. Assumes that the train and"
        + "validation datasets are one level up from repo root. Also assumes that"
        + "the train and validation splits are present in the experiment directory")

    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well."
        + "Note: Keep the train, val, test split json files in this directory.",
    )

    arg_parser.add_argument(
        "--model",
        "-m",
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    arg_parser.add_argument(
        "--val",
        action='store_true',
        help="Whether to compute the metrics on validation set",
    )

    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        default="latest",
        help="The model checkpoint to use.",
    )

    args = arg_parser.parse_args()

    YCB_MODELS_ALL = [
    "003_cracker_box_google_16k_textured_scale_1000",
    "005_tomato_soup_can_google_16k_textured_scale_1000",
    "006_mustard_bottle_google_16k_textured_scale_1000",
    "007_tuna_fish_can_google_16k_textured_scale_1000",
    "008_pudding_box_google_16k_textured_scale_1000",
    "009_gelatin_box_google_16k_textured_scale_1000",
    "010_potted_meat_can_google_16k_textured_scale_1000",
    "021_bleach_cleanser_google_16k_textured_scale_1000"
    ]

    if not os.path.isdir(args.experiment_directory):
        print("The specified experiment directory does not exist!")
        print(args.experiment_directory)
        exit(0)
    
    if args.model not in YCB_MODELS_ALL:
        print("The specified model name (f{args.model}) is not correct. Please check:")
        print(YCB_MODELS_ALL)
        exit(0)

    ycb_id = args.model.split('_')[0]
    if ycb_id not in args.experiment_directory:
        print("There is a mismatch between the experiment dir and ycb model!")
        print("YCB ID and Model: ", ycb_id, args.model)
        exit(0)

    main_function(args.experiment_directory, args.model, args.checkpoint, args.val)
