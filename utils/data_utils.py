import os
import random
import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform
import trimesh
import torch
import torch.utils.data

import utils.misc as misc_utils

def construct_cmap_array(contactmap_list):
    if not contactmap_list:
        logging.error("Provide a non-empty list for the contactmaps!")
        return
    # Create a (N,d) arrray of individual contactmaps
    # But we might not know "d" before hand.
    # So load the first file and check it out
    _dummy_data = np.load(contactmap_list[0])['heatmap']
    # print(_dummy_data.shape)
    N = len(contactmap_list)
    d = _dummy_data.shape[0]
    contact_maps = np.empty((N, d))
    for i, npf in enumerate(contactmap_list):
        contact_maps[i] = np.load(npf)['heatmap']
    return contact_maps

def construct_cmap_dist(contactmap_list):
    contact_maps = construct_cmap_array(contactmap_list)
    D = squareform(pdist(contact_maps, metric='cityblock'))
    # Normalize to [0,1]
    D /= np.max(D)
    return D


def get_gripper_urdf_pc(gripper_urdf_pc_file):
    npz = np.load(gripper_urdf_pc_file)
    points = npz['point_data']
    return torch.from_numpy(points).float()


def get_surface_points_neg(npzfile, gripper=True, samples=500, surface_dist=0.005, data=None):
    if npzfile:
        npz = np.load(npzfile)
    else:
        npz = data
    try:
        sdf_data = npz['data']
        if gripper:
            # gripper surface
            neg_data = sdf_data[sdf_data[:, 4] <= 0]
        else:
            # object surface
            neg_data = sdf_data[sdf_data[:, 3] <= 0]
        neg_tensor = remove_nans(torch.from_numpy(neg_data))
    except Exception as e:
        print(f"fail to load {npzfile}, {e}")
    neg_tensor = remove_higher_dist(neg_tensor, surface_dist)
    random_neg = (torch.rand(samples) * neg_tensor.shape[0]).long()
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    return sample_neg


def filter_invalid_sdf_values(tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (
        torch.abs(tensor[:, 4]) < abs(dist))
    # print(keep[:15])
    return tensor[keep, :]


def get_instance_namelist(data_source: str, split_dict: dict) -> list:
    npzfiles = []
    bad_filepath_count = 0
    # one file per grasping "scene" i.e gripper and object interaction
    # Each entry has the format: "datadir/object/sdf/gripper/filenum.npz"
    sdf_dir = misc_utils.data_sdf_dir

    # def is_valid_file(datadir, object, sdfdir, gripper, f):
    #     fname = os.path.join(datadir, dataroot_relative_path(object, sdfdir, gripper, f))
    #     check = os.path.isfile(fname)
    #     if not check:
    #         logging.warning(f"Requested non-existent npz file {fname}")
    #     return check

    def full_path(datadir, object, sdfdir, gripper, f):
        return os.path.join(datadir, object, sdfdir, gripper, f)

    for dataset in split_dict:
        dset_dict = split_dict[dataset]
        for object_name in dset_dict:
            obj_dict = dset_dict[object_name]
            for gripper in obj_dict:
                file_list = obj_dict[gripper]
                for f in file_list:
                    fpath = full_path(
                        data_source, object_name, sdf_dir, gripper, f)
                    if not os.path.isfile(fpath):
                        logging.warning(
                            f"Requested non-existent npz file {fpath}")
                        bad_filepath_count += 1
                    else:
                        npzfiles.append(f)
    logging.warning(f"Non-existent file count: {bad_filepath_count}")
    # NOT DOING ANYTHING WITH THE NORMALIZATION PARAMS
    return npzfiles


def get_instance_filelist(data_source: str, split_dict: dict) -> tuple:
    npzfiles = []
    # Each npzfile will also have a corresponding entry in gripper_pc_files as to which
    # is the gripper point cloud file (in its canonical urdf config)
    # Useful for dealing with multi-gripper inputs
    gripper_pc_files = []
    gripper_names = []
    contactmap_files = []

    bad_filepath_count = 0
    # one file per grasping "scene" i.e gripper and object interaction
    # Each entry has the format: "datadir/object/sdf/gripper/filenum.npz"
    sdf_dir = misc_utils.data_sdf_dir
    urdf_pc_dir = misc_utils.data_urdf_pc
    contactmap_dir = misc_utils.datadir_contactmap

    for dataset in split_dict:
        dset_dict = split_dict[dataset]
        for object_name in dset_dict:
            obj_dict = dset_dict[object_name]
            for gripper in obj_dict:
                file_list = obj_dict[gripper]
                for f in file_list:
                    fpath = os.path.join(
                        data_source, object_name, sdf_dir, gripper, f)
                    # For cmap_file, disregard the 'sdf_' part of string in variable f
                    # and merge with 'contactmap_' part.
                    cmap_file = 'contactmap_' + '_'.join(f.split('_')[1:])
                    cmap_fpath = os.path.join(
                        data_source, object_name, contactmap_dir, gripper, cmap_file)
                    if not os.path.isfile(cmap_fpath):
                        logging.warning(
                            f"Requested non-existent npz file {cmap_fpath}")
                        bad_filepath_count += 1
                    else:
                        contactmap_files.append(cmap_fpath)

                    if not os.path.isfile(fpath):
                        logging.warning(
                            f"Requested non-existent npz file {fpath}")
                        bad_filepath_count += 1
                    else:
                        npzfiles.append(fpath)
                        gripper_pc_files.append(os.path.join(data_source, urdf_pc_dir,
                                                             gripper + ".npz"))
                        gripper_names.append(gripper)

    if bad_filepath_count > 0:
        logging.warning(f"Non-existent file count: {bad_filepath_count}")

    # NOT DOING ANYTHING WITH THE NORMALIZATION PARAMS
    return contactmap_files, gripper_names, gripper_pc_files, npzfiles


def get_sdf_samples(fname, samples_per_scene=None, filter_dist=False):
    npz = np.load(fname)
    if not samples_per_scene:
        return torch.from_numpy(npz['data'])

    sdf_data = torch.from_numpy(npz['data'])
    # problem with this is that same point may come again!
    pos_obj = sdf_data[sdf_data[:, 3] > 0]
    pos_grp = sdf_data[sdf_data[:, 4] > 0]

    neg_obj = sdf_data[sdf_data[:, 3] <= 0]
    neg_grp = sdf_data[sdf_data[:, 4] <= 0]

    pos_tensor = torch.cat([pos_obj, pos_grp], 0)
    neg_tensor = torch.cat([neg_obj, neg_grp], 0)

    if filter_dist:
        pos_tensor = filter_invalid_sdf_values(pos_tensor, 1.0)
        neg_tensor = filter_invalid_sdf_values(neg_tensor, 1.0)

    # split the sample into half
    half_samples = int(samples_per_scene / 2)
    random_pos = (torch.rand(half_samples) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half_samples) * neg_tensor.shape[0]).long()

    samples_pos = torch.index_select(pos_tensor, 0, random_pos)
    samples_neg = torch.index_select(neg_tensor, 0, random_neg)
    samples = torch.cat([samples_pos, samples_neg], 0)
    return samples


def unpack_sdf_samples_from_ram(sdf_data, subsample=None):
    if not subsample:
        return sdf_data

    # problem with this is that same point may come again!
    pos_obj = sdf_data[sdf_data[:, 3] > 0]
    pos_grp = sdf_data[sdf_data[:, 4] > 0]

    neg_obj = sdf_data[sdf_data[:, 3] <= 0]
    neg_grp = sdf_data[sdf_data[:, 4] <= 0]

    pos_tensor = torch.cat([pos_obj, pos_grp], 0)
    neg_tensor = torch.cat([neg_obj, neg_grp], 0)

    # split the sample into half
    half_samples = int(subsample / 2)
    random_pos = (torch.rand(half_samples) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half_samples) * neg_tensor.shape[0]).long()

    samples_pos = torch.index_select(pos_tensor, 0, random_pos)
    samples_neg = torch.index_select(neg_tensor, 0, random_neg)
    samples = torch.cat([samples_pos, samples_neg], 0)
    return samples


def read_sdf_samples_into_ram(filename):
    return get_sdf_samples(filename)


##########################################################################
# Deep SDF Code
##########################################################################

def dsdf_get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(
                        data_source, misc_utils.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(
                            instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


def dsdf_unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind: (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind: (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def dsdf_read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    return [pos_tensor, neg_tensor]

##########################################################################
# Grasping Field Code
##########################################################################


def load_points(filename):
    points = []
    # print(filename)
    with open(filename, 'r') as fp:
        for line in fp:
            point = line.strip().split(" ")[1:]
            point = np.asarray(point)
            point = point.astype(float)
            points.append(point)
    return np.asarray(points)


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def remove_higher_dist(tensor, dist, both_col=False):
    keep = torch.abs(tensor[:, 3]) < abs(dist)
    return tensor[keep, :3]


def filter_invalid_sdf(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (
        torch.abs(tensor[:, 4]) < abs(dist))
    # print(keep[:15])
    return tensor[keep, :], lab_tensor[keep, :]


def unpack_normal_params(filename):
    npz = np.load(filename)
    scale = torch.from_numpy(npz["scale"])
    offset = torch.from_numpy(npz["offset"])
    return scale, offset


def unpack_sdf_samples(filename, subsample=None, hand=True, clamp=None, filter_dist=False):
    npz = np.load(filename)
    if subsample is None:
        return npz
    try:
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
        pos_sdf_other = torch.from_numpy(npz["pos_other"])
        neg_sdf_other = torch.from_numpy(npz["neg_other"])
        if hand:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
        else:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
    except Exception as e:
        print("fail to load {}, {}".format(filename, e))
    # make it (x,y,z,sdf_to_hand,sdf_to_obj)
    if hand:
        pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
        neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
    else:
        xyz_pos = pos_tensor[:, :3]
        sdf_pos = pos_tensor[:, 3].unsqueeze(1)
        pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

        xyz_neg = neg_tensor[:, :3]
        sdf_neg = neg_tensor[:, 3].unsqueeze(1)
        neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

    # split the sample into half
    half = int(subsample / 2)

    if filter_dist:
        pos_tensor, lab_pos_tensor = filter_invalid_sdf(
            pos_tensor, lab_pos_tensor, 2.0)
        neg_tensor, lab_neg_tensor = filter_invalid_sdf(
            neg_tensor, lab_neg_tensor, 2.0)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    # label
    sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
    sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

    # hand part label
    # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
    hand_part_pos = sample_pos_lab[:, 0]
    hand_part_neg = sample_neg_lab[:, 0]
    samples = torch.cat([sample_pos, sample_neg], 0)
    labels = torch.cat([hand_part_pos, hand_part_neg], 0)

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1
        # print(labels)

    if not hand:
        labels[:] = -1
    return samples, labels


def gf_get_instance_filenames(data_source, input_type, encoder_input_source, split, check_file=True, fhb=False, dataset_name="obman"):
    unusable_count = 0
    npzfiles_hand = []
    npzfiles_obj = []
    normalization_params = []
    # imagefiles = []
    encoder_input_files = []
    for dataset in split:
        for class_name in split[dataset]:
            # Hand
            hand_instance_filename = os.path.join(
                dataset, class_name, "hand.npz")
            if check_file and not os.path.isfile(
                os.path.join(
                    data_source, misc_utils.sdf_samples_subdir, hand_instance_filename)
            ):
                logging.warning(
                    "Requested non-existent hand file '{}'".format(
                        hand_instance_filename)
                )
                unusable_count += 1
                continue

            # Object
            obj_instance_filename = os.path.join(
                dataset, class_name, "obj.npz")
            if check_file and not os.path.isfile(
                os.path.join(
                    data_source, misc_utils.sdf_samples_subdir, obj_instance_filename)
            ):
                logging.warning(
                    "Requested non-existent object file '{}'".format(
                        obj_instance_filename)
                )
                unusable_count += 1
                continue

            # Offset and scale
            normalization_params_filename = os.path.join(
                dataset, class_name, "obj.npz")
            if check_file and not os.path.isfile(
                os.path.join(
                    data_source, misc_utils.normalization_param_subdir, normalization_params_filename)
            ):
                logging.warning(
                    "Requested non-existent normalization params file '{}'".format(
                        normalization_params_filename)
                )
                unusable_count += 1
                continue

            if input_type == 'point_cloud':
                encoder_input_files = []
                pass

            npzfiles_hand += [hand_instance_filename]
            npzfiles_obj += [obj_instance_filename]
            normalization_params += [normalization_params_filename]

    logging.warning(
        "Non-existent file count: {} out of {}".format(
            unusable_count, len(npzfiles_hand))
    )
    return npzfiles_hand, npzfiles_obj, normalization_params, encoder_input_files


def get_negative_surface_points(filename=None, pc_sample=500, surface_dist=0.005, data=None):

    if filename:
        npz = np.load(filename)
    else:
        npz = data
    try:
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    except Exception as e:
        print("fail to load {}, {}".format(filename, e))

    neg_tensor = remove_higher_dist(neg_tensor, surface_dist)

    random_neg = (torch.rand(pc_sample) * neg_tensor.shape[0]).long()
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    return sample_neg


def normalize_obj_center(encoder_input_hand, encoder_input_obj, hand_samples=None, obj_samples=None, scale=1.0):
    object_center = encoder_input_obj.mean(dim=0)
    encoder_input_hand = encoder_input_hand - object_center
    encoder_input_obj = encoder_input_obj - object_center
    if (hand_samples is not None) and (obj_samples is not None):
        hand_samples[:, :3] = hand_samples[:, :3] - object_center / scale
        obj_samples[:, :3] = obj_samples[:, :3] - object_center / scale
        return encoder_input_hand, encoder_input_obj, hand_samples, obj_samples
    return encoder_input_hand, encoder_input_obj


##########################################################################
# Simpler Models
##########################################################################
# This works for a single sdf prediction network


def get_weights(labels):
    N = len(labels)
    weight_negative = float((labels >= 0).sum())/N
    return [weight_negative if l < 0 else 1-weight_negative for l in labels]

# With two sdf predictions, "positive" examples are points such that they are outside
# both the gripper and object


def get_weights_dual(sdf_obj, sdf_grp):
    N = len(sdf_obj)
    both_pos = (np.sign(sdf_obj) == 1) & (np.sign(sdf_grp) == 1)
    count_both_pos = np.sum(both_pos)
    weight_negative = count_both_pos * 1.0/N
    weight_positive = 1.0 - weight_negative
    return [weight_negative if both_pos[i] else weight_positive for i in range(N)]


# Hacky function. Edit later
# Returns the Nx5 array with query points and sdfs to obj and gripper
def load_from_npz(datadir, obj, gripper, graspnum_idx):
    sdf_fname = f'sdf_graspnum_{graspnum_idx}.npz'
    fpath = os.path.join(datadir, obj, 'sdf', gripper, sdf_fname)
    data = np.load(fpath)
    return data['data']


def train_val_split(points_sdf_data, val_fraction=0.3):
    N = points_sdf_data.shape[0]
    val_mask = np.zeros(N, dtype=np.bool)
    val_ind = np.random.choice(range(N), int(val_fraction*N))
    val_mask[val_ind] = 1

    data_val = points_sdf_data[val_mask]
    data_trn = points_sdf_data[~val_mask]
    return data_trn, data_val


def format_data(points_sdf_data):
    xyz_points = points_sdf_data[:, :3]
    sdf_obj = points_sdf_data[:, 3]
    sdf_obj = np.expand_dims(sdf_obj, -1)
    sdf_grp = points_sdf_data[:, 4]
    sdf_grp = np.expand_dims(sdf_grp, -1)
    return xyz_points, sdf_obj, sdf_grp
