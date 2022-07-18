import torch
import numpy as np
import os
import json
import sys
sys.path.append('..')
import utils.dataset
from utils.eval_utils import extract_graspidx_from_npzfile

def get_graspit_grasps(experiment_dir, dataset_dir, object_model, train_split_file='split_train.json') -> list[dict]:
    '''
    Takes in an experiment run (experiment_dir, dataset_dir, train_split) and returns
    the corresponding graspit grasps that were used for the training. It returns a list[dict]
    `graspit_grasps` with length = total number of training samples. For e.g. with 5 
    grippers, and 50 grasps  each, it will return a list with len=250 so that there is 1-1
    correspondence between the indexing in latent vectors and graspit grasps.

    Input:
        experiment_dir: Path to the experiment dir which hosts the specs.json and 
                        split_train.json

        dataset_dir: Path to the dataset root which contains the `refined_grasps` folder

        train_split: JSON file containing the training data used (Optional) -- this should
                     be picked up from the provided experiments dir.

    Returns:
        graspit_grasps (list[dict]): A list of len training data samples with a dict for
                                     for each, containing 'gripper', 'pose' and 'dofs' for
                                     the corresponding grasp.
    '''

    train_split_fpath = os.path.join(experiment_dir, train_split_file)
    with open(train_split_fpath, "r") as f:
        train_split = json.load(f)

    sdf_dataset = utils.dataset.MultiGripperSamples(dataset_dir, train_split,
                                                    subsample=10000)

    gripper_names = sdf_dataset.grippers_list
    print(
        f"There were total {len(gripper_names)} grippers used with the idxs:")
    print(gripper_names)
    # print(sdf_dataset.grp_idxs)

    print("Loading the graspit grasp json files for each gripper")
    gripper_graspit_info = {}
    for grp in gripper_names:
        grasps_file = f'refined_{object_model}-{grp}.json'
        grasps_path = os.path.join(dataset_dir, 'refined_grasps', grasps_file)
        with open(grasps_path, 'r') as gf:
            all_data = json.load(gf)
        assert grp == all_data['gripper']
        # object_model is with the "_scale_1000" while the object_id is not
        # Hence the splicing to exclude that part
        assert object_model[:-11] == all_data['object_id']
        # Just store the grasps here.
        gripper_graspit_info[grp] = all_data['grasps']

    print("Storing the graspit grasps for training data sample")
    graspit_grasps = [{} for _ in range(len(sdf_dataset))]
    for i in range(len(sdf_dataset)):
        grp_idx, _, _, npzfile = sdf_dataset[i]
        gripper = gripper_names[grp_idx]
        grasp_index = extract_graspidx_from_npzfile(npzfile)
        grasp_i = gripper_graspit_info[gripper][grasp_index]
        graspit_grasps[i]['gripper'] = gripper
        graspit_grasps[i]['pose'] = grasp_i['pose']
        graspit_grasps[i]['dofs'] = grasp_i['dofs']

    return graspit_grasps

def main():
    # Testing out the functionality of the code.

    exp_dir = '../experiments/all5_003_dsdf_50_varcmap/'
    # data_dir = '/home/ninad/Desktop/multi-finger-grasping/output_dataset'
    data_dir = '../docker-data/output_dataset'    
    obj_model = "003_cracker_box_google_16k_textured_scale_1000"

    gg = get_graspit_grasps(exp_dir, data_dir, obj_model)


if __name__ == "__main__":
    main()
