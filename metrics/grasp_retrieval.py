import argparse
import json
import os
import random
import pickle

import torch
import numpy as np

from scipy.spatial.distance import pdist, cdist, squareform

from PIL import Image
from PIL.Image import Image as PilImage
import matplotlib.pyplot as plt

import sys 

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

THIS_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(THIS_DIR, '..')
add_path(LIB_PATH)

import utils.misc as ws
import utils.data_utils
import utils.train_utils
import utils.eval_utils
import utils.mesh
import utils.dataset

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

def get_actual_idx(training_contact_maps, global_idx):
    _path_split = training_contact_maps[global_idx].split("/")
    return (_path_split[-2], _path_split[-1].split(".")[0].split("_")[-1])

def check_matches(cmap_dist, lv_dist, val=False, extreme_end=5):
    count_top1 = 0
    count_topK = 0
    N = len(cmap_dist)
    for q in range(N):
        topK_query_cmap = np.argsort(cmap_dist[q])
        topK_query_lv = np.argsort(lv_dist[q])
        # print(topK_query_cmap.shape, topK_query_lv.shape)

        if val:
            # top-1
            if topK_query_cmap[0] == topK_query_lv[0]:
                count_top1 += 1
            if topK_query_cmap[0] in topK_query_lv[:extreme_end]:
                count_topK += 1
        else:
            # top-1
            if topK_query_cmap[1] == topK_query_lv[1]:
                count_top1 += 1
            # top-5 -- [:6] since first element is the same as query!
            if topK_query_cmap[1] in topK_query_lv[1:extreme_end+1]:
                count_topK += 1

    return count_top1/N, count_topK/N

def check_avg_sim(cmap_dist, lv_dist, cmap_sim, val=False, K=1):
    sim_topK = 0
    sim_cmap = 0
    far_topK = 0
    far_cmap = 0
    N = len(cmap_dist) # number of validation samples
    for q in range(N):
        topK_query_cmap = np.argsort(cmap_dist[q])
        topK_query_lv = np.argsort(lv_dist[q])
        if val:
            sim_topK += np.mean(cmap_sim[q][topK_query_lv[:K]])
            sim_cmap += np.mean(cmap_sim[q][topK_query_cmap[:K]])
        else: # 1 since 0 corresponds to the same training example
            sim_topK += np.mean(cmap_sim[q][topK_query_lv[1:K+1]])
            sim_cmap += np.mean(cmap_sim[q][topK_query_cmap[1:K+1]]    )
        # Farthest (same code for both)
        far_topK += np.mean(cmap_sim[q][topK_query_lv[-K:]])
        far_cmap += np.mean(cmap_sim[q][topK_query_cmap[-K:]])
    return sim_topK/N, sim_cmap/N, far_topK/N, far_cmap/N


def main_function(experiments_dir, ycb_model, validation=False):    
    print("===============================================================")
    print(f"metric_GraspRetrieval-IsVal_{validation}-{ycb_model}")

    dataset_dir = os.path.join(LIB_PATH, '../dataset_train/')
    validation_dir = os.path.join(LIB_PATH, '../dataset_validation/')
    # trn_cmap_dir = os.path.join(dataset_dir, ycb_model, "contactmap")
    # val_cmap_dir = os.path.join(validation_dir, ycb_model, "contactmap")
    trn_imgs_dir = os.path.join(dataset_dir, ycb_model, "images")
    val_imgs_dir = os.path.join(validation_dir, ycb_model, "images")

    CHECKPOINT = 'latest'
    LATENT_CODE_DIR = ws.latent_codes_subdir

    trn_split_file = os.path.join(experiments_dir, 'split_train.json')
    val_split_file = os.path.join(experiments_dir, 'split_validation.json')
    specs_filename = os.path.join(experiments_dir, "specs.json")
    with open(trn_split_file, 'r') as f:
        trn_data_split = json.load(f)
    with open(val_split_file, 'r') as f:
        val_data_split = json.load(f)
    
    saved_model_state = torch.load(
        os.path.join(
            experiments_dir, ws.model_params_subdir, CHECKPOINT + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    # Load the Training data set
    trn_cmap_f, trn_grp_names, trn_gpc_f, trn_npz_filenames = utils.data_utils.get_instance_filelist(
        dataset_dir, trn_data_split)
    trn_cmaps = utils.data_utils.construct_cmap_array(trn_cmap_f)
    # print(f"Training cmaps loaded. Shape: {trn_cmaps.shape}")

    # Load the training latent vectors
    specs = json.load(open(specs_filename))
    latent_size = specs["CodeLength"]
    trn_latent_vecs = ws.load_latent_vectors(experiments_dir, CHECKPOINT)
    # print(latent_size, trn_latent_vecs.shape)
    trn_lv = trn_latent_vecs.cpu().numpy()
    # print(f"Training LVs loaded. Shape: {trn_lv.shape}")
    
    results_dict = {}
    results_dict['ycb_model'] = ycb_model
    results_dict['is_val'] = validation
    topK_vals = [1,3,5, 10]
    if not validation:        
        # Create distance matrix between TRN and TRN Contact Maps
        cmapdist_trn_to_trn = squareform(pdist(trn_cmaps, metric='cityblock'))
        cmapdist_trn_to_trn /= np.max(cmapdist_trn_to_trn)
        cmapsim_trn_to_trn = 1 - cmapdist_trn_to_trn

        # Create distance matrix between TRN and TRN Latent Vectors
        lvdist_trn_to_trn = squareform(pdist(trn_lv, metric='cityblock'))
        lvdist_trn_to_trn /= np.max(lvdist_trn_to_trn)
        # lvsim_trn_to_trn = 1 - lvdist_trn_to_trn
        print("Similarity Metrics on TRAINING")
        print("Nearest (from LV) | Nearest from G.T Cmap | Farthest from LV | Farthest from G.T Cmap")
        for k in topK_vals:
            results_dict[k] = check_avg_sim(
                cmapdist_trn_to_trn, lvdist_trn_to_trn, cmapsim_trn_to_trn, validation, k)
            print("K=", k, results_dict[k])

        # print("K=1", check_avg_sim(cmapdist_trn_to_trn, lvdist_trn_to_trn, cmapsim_trn_to_trn, 1))
        # print("K=3", check_avg_sim(cmapdist_trn_to_trn, lvdist_trn_to_trn, cmapsim_trn_to_trn, 3))
        # print("K=5", check_avg_sim(cmapdist_trn_to_trn, lvdist_trn_to_trn, cmapsim_trn_to_trn, 5))
        
        # print("k: ", 2, check_matches(cmapdist_trn_to_trn, lvdist_trn_to_trn, validation, extreme_end=2))
        # print("k: ", 3, check_matches(cmapdist_trn_to_trn, lvdist_trn_to_trn, validation, extreme_end=3))
        # print("k: ", 5, check_matches(cmapdist_trn_to_trn, lvdist_trn_to_trn, validation, extreme_end=5))
        # print("k: ", 8, check_matches(cmapdist_trn_to_trn, lvdist_trn_to_trn, validation, extreme_end=8))
        # print("k: ", 10, check_matches(cmapdist_trn_to_trn, lvdist_trn_to_trn, validation, extreme_end=10))
    else:        
        # LOADS THE VALIDATION DATA SET
        val_cmap_f, val_grp_names, _, val_npz_filenames = utils.data_utils.get_instance_filelist(
            validation_dir, val_data_split)
        val_cmaps = utils.data_utils.construct_cmap_array(val_cmap_f)
        # print(f"Validation cmaps loaded. Shape: {val_cmaps.shape}")
        
        # Load the VAL latent vectors
        N = len(val_npz_filenames)
        val_lv = np.empty((N, latent_size))
        for idx in range(len(val_npz_filenames)):
            gripper = val_grp_names[idx]
            npz = val_npz_filenames[idx]
            # full_filename = npz
            # print(idx, npz[-35:])
            npz_instance_name = os.path.split(npz)[-1].split('.')[0]
            
            lvec_file = get_reconstructed_code_filename(
                experiments_dir, saved_model_epoch,
                ycb_model, gripper, npz_instance_name)
            
            lvec_idx = torch.load(lvec_file)
            lvec_idx = lvec_idx.squeeze()
            val_lv[idx] = lvec_idx.cpu().detach().numpy()
        # print(f"Validation LVs loaded. Shape: {val_lv.shape}")
        
        # Create distance matrix between VAL and TRN Contact Maps
        lvdist_val_to_trn = cdist(val_lv, trn_lv, metric='cityblock')
        lvdist_val_to_trn /= np.max(lvdist_val_to_trn)
        lvsim_val_to_trn = 1 - lvdist_val_to_trn
        # print(f"Dist matrix over LVs val -> trn construced. Shape: {lvdist_val_to_trn.shape}")        
        
        cmapdist_val_to_trn = cdist(val_cmaps, trn_cmaps, metric='cityblock')
        cmapdist_val_to_trn /= np.max(cmapdist_val_to_trn)
        cmapsim_val_to_trn = 1 - cmapdist_val_to_trn
        # print(f"Dist matrix over Cmaps val -> trn construced. Shape: {cmapdist_val_to_trn.shape}")
        print("Similarity Metrics on VALIDATION")
        print("Nearest (from LV) | Nearest from G.T Cmap | Farthest from LV | Farthest from G.T Cmap")
        for k in topK_vals:
            results_dict[k] = check_avg_sim(
                cmapdist_val_to_trn, lvdist_val_to_trn, cmapsim_val_to_trn, validation, k)
            print("K=", k, results_dict[k])
        
        # print("K=1", check_avg_sim(cmapdist_val_to_trn, lvdist_val_to_trn, cmapsim_val_to_trn, 1))
        # print("K=3", check_avg_sim(cmapdist_val_to_trn, lvdist_val_to_trn, cmapsim_val_to_trn, 3))
        # print("K=5", check_avg_sim(cmapdist_val_to_trn, lvdist_val_to_trn, cmapsim_val_to_trn, 5))

        # print("k: ", 2, check_matches(cmapdist_val_to_trn, lvdist_val_to_trn, validation, extreme_end=2))
        # print("k: ", 3, check_matches(cmapdist_val_to_trn, lvdist_val_to_trn, validation, extreme_end=3))
        # print("k: ", 5, check_matches(cmapdist_val_to_trn, lvdist_val_to_trn, validation, extreme_end=5))
        # print("k: ", 8, check_matches(cmapdist_val_to_trn, lvdist_val_to_trn, validation, extreme_end=8))
        # print("k: ", 10, check_matches(cmapdist_val_to_trn, lvdist_val_to_trn, validation, extreme_end=10))
    result_file = f"metric_GraspRetrieval-isVal_{validation}-{ycb_model}.p"
    with open(os.path.join('./', 'results', result_file), "wb") as f:
        pickle.dump(results_dict, f)
    print("===============================================================\n")

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

    main_function(args.experiment_directory, args.model, args.val)