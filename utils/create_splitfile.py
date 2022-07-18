import json
import argparse
from json.tool import main
import os
import math

import numpy as np

import misc as misc_utils


def save_json(fpath, data):
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=2)


parser = argparse.ArgumentParser(
    description='Script to generate the train,val,test split files. It will output the split files to the root of experiments directory')

parser.add_argument('-d', '--dataset_dir', type=str, default='/home/ninad/Desktop/Docs/phd-res/proj-irvl-grasp-transfer/code/docker-data/output_dataset/',
                    help='dataset directory where the point clouds will be populated')
# Data set directory. Assumes that:
# The grasps will be in: datadir/refined_grasps/obj-gripper.json
# The saved point clouds will be in: datadir/object_name/point_cloud/gripper/*.ply

parser.add_argument('-m', '--models', nargs='*',
                    default=['003_cracker_box_google_16k_textured_scale_1000'])

parser.add_argument('-l', '--models_file', type=str,
                    default='')  # one ycb object model entry per line

parser.add_argument('-g', '--grippers', nargs='*',
                    default=['fetch_gripper', 'Barrett', 'HumanHand'])

# Whether to create a toy dataset or not
parser.add_argument('--create_toy',
                    default=False, action='store_true', 
                    help='Creates a small toy dataset with 1-2 samples for debugging')

# Whether to use the entire dataset for training
parser.add_argument('--create_full',
                    default=False, action='store_true', 
                    help='Creates a dataset with all the samples i.e no validation data')


def main_function(args):
    dataset_name = 'ycb-combinedsdf'
    
    # data_source = '/home/ninad/Desktop/Docs/phd-res/proj-irvl-grasp-transfer/code/docker-data/output_dataset/'
    # object_list = ['003_cracker_box_google_16k_textured_scale_1000']
    # gripper_list = ['fetch_gripper', 'Barrett', 'HumanHand']
    # # gripper_list = ['fetch_gripper']

    if not args.dataset_dir:
        print('Output directory not specified')
        exit(0)
    else:
        if os.path.isdir(args.dataset_dir):
            data_source = args.dataset_dir
        else:
            print(f"Invalid dataset dir: {args.dataset_dir}!!\n")
            print("EXITING ....")
            exit(0)

    if args.models:
        object_list = args.models
    elif args.models_file:
        with open(args.models_file) as f:
            object_list = f.read().splitlines()
    else:
        print("Neither models nor models file specified!")
        exit(0)

    if args.create_toy:
        num_trn, num_val, num_tst = 2, 2, 2
    elif args.create_full:
        print("Using the entire set for training!")
    else:
        trn_fraction, val_fraction, tst_fraction = 0.8, 0.1, 0.1
        num_trn, num_val, num_tst = None, None, None  

    gripper_list = args.grippers

    sdf_dir = misc_utils.data_sdf_dir

    split_dict_trn = {}
    split_dict_val = {}
    split_dict_tst = {}

    split_dict_trn[dataset_name] = {}
    split_dict_val[dataset_name] = {}
    split_dict_tst[dataset_name] = {}

    for object_model in object_list:
        obj_dict_trn = {}
        obj_dict_val = {}
        obj_dict_tst = {}
        for gripper in gripper_list:
            dir_files = os.path.join(
                data_source, object_model, sdf_dir, gripper)
            # Complicated lambda function so that '10.npz' comes after '2.npz' i.e natural sort
            file_list = sorted(os.listdir(dir_files), key=lambda x: int(
                os.path.splitext(x.split('_')[-1])[0]))
            N = len(file_list)
            
            if args.create_toy:
                idx_trn = 2
                idx_val = idx_trn + 2
                idx_tst = idx_val + 2
            elif args.create_full:
                idx_trn = N
            else:            
                idx_trn = math.floor(N * trn_fraction)
                idx_val = idx_trn + math.floor(N * val_fraction)
                idx_tst = N-1
            
            obj_dict_trn[gripper] = file_list[:idx_trn]
            if not args.create_full:
                obj_dict_val[gripper] = file_list[idx_trn: idx_val]
                obj_dict_tst[gripper] = file_list[idx_val:idx_tst]

        split_dict_trn[dataset_name][object_model] = obj_dict_trn
        if not args.create_full:
            split_dict_val[dataset_name][object_model] = obj_dict_val
            split_dict_tst[dataset_name][object_model] = obj_dict_tst

    trn_split_file = '../experiments/split_train.json'
    save_json(trn_split_file, split_dict_trn)
    
    if not args.create_full:
        val_split_file = '../experiments/split_validation.json'
        tst_split_file = '../experiments/split_test.json'

        save_json(val_split_file, split_dict_val)
        save_json(tst_split_file, split_dict_tst)


if __name__ == '__main__':
    main_function(parser.parse_args())
