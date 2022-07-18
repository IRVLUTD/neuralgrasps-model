import enum
import os
import logging

import numpy as np
import trimesh
from skimage import io, transform, color

import torch
import torch.utils.data
from torchvision import transforms

import utils.misc as misc_utils
import utils.data_utils as data_utils


class MultiGripperSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split_dict,
        subsample,
        filter_dist=False,
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.filter_dist = filter_dist

        self.cmaps, self.grp_names, self.gripper_pc_files, self.npzfiles = data_utils.get_instance_filelist(
            data_source, split_dict)

        logging.debug(
            "using "
            + str(len(self.npzfiles))
            + " grasps from data source "
            + data_source
        )

        # Count the number of grippers:
        grippers = set()
        for dataset in split_dict:
            dset_dict = split_dict[dataset]
            for object_name in dset_dict:
                obj_dict = dset_dict[object_name]
                grippers.update(obj_dict.keys())

        grippers_set = set()
        for grp in self.grp_names:
            grippers_set.add(grp)
        grippers_list = list(grippers_set)
        grippers_list.sort()
        gripper_with_idx = {k: v for v, k in enumerate(grippers_list)}

        self.num_grippers = len(grippers_list)
        # This will be the index into the pytorch embedding layer
        self.grippers_list = grippers_list
        self.grp_idxs = [gripper_with_idx[grp_name]
                         for grp_name in self.grp_names]
        print(f"gripper idxs: {gripper_with_idx}")
        self.gripper_with_idx = gripper_with_idx  # For later retrieval

        # Construct the contactmap matrix and load it in memorys
        self.cmap_dist = data_utils.construct_cmap_dist(self.cmaps)

    def __len__(self):
        return len(self.npzfiles)

    def __getitem__(self, idx):
        npzfile = self.npzfiles[idx]
        gripper_pc_file = self.gripper_pc_files[idx]
        grp_idx = self.grp_idxs[idx]
        num_sample = self.subsample

        samples = data_utils.get_sdf_samples(
            npzfile, num_sample, self.filter_dist)
        # NOT USING THE GRIPPER PC FOR CoRL
        # gripper_pc = data_utils.get_gripper_urdf_pc(gripper_pc_file)  # 4096, 3
        # gripper_pc = None
        # gripper_pc = gripper_pc.repeat_interleave(self.subsample, dim=0)
        return grp_idx, samples, idx, npzfile


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split_dict,
        subsample,
        filter_dist=False,
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.filter_dist = filter_dist

        _, _, _, self.npzfiles = data_utils.get_instance_filelist(
            data_source, split_dict)
        logging.debug(
            "using "
            + str(len(self.npzfiles))
            + " grasps from data source "
            + data_source
        )

    def __len__(self):
        return len(self.npzfiles)

    def __getitem__(self, idx):
        npzfile = self.npzfiles[idx]
        num_sample = self.subsample
        samples = data_utils.get_sdf_samples(
            npzfile, num_sample, self.filter_dist)
        return samples, idx, npzfile


######################################################################################
############### Grasping Field Code ##################################################
######################################################################################
class PointCloudInput(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        data_list,
        sample_surface=False,
        pc_sample=500,
        verbose=0,
        model_type="1encoder1decoder",
    ):
        self.data_source = data_source
        # self.imagefiles = get_images_filenames(image_source, split, fhb=fhb)

        self.sample_surface = sample_surface
        self.pc_sample = pc_sample

        self.input_files = data_list

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):

        filename = os.path.join(self.data_source, self.input_files[idx])
        # print("load mesh", filename)

        if self.sample_surface:
            global_scale = 5.0
            input_mesh = trimesh.load(filename, process=False)
            surface_points = trimesh.sample.sample_surface(
                input_mesh, self.pc_sample)[0]
            surface_points = torch.from_numpy(
                surface_points * global_scale).float()
        else:
            surface_points = torch.from_numpy(
                data_utils.load_points(filename)).float()
        # print(surface_points)

        return surface_points, idx, self.input_files[idx]


class SDFSamples_old(torch.utils.data.Dataset):
    def __init__(
        self,
        input_type,
        data_source,
        split,
        subsample,
        dataset_name="obman",
        image_source=None,
        hand_branch=True,
        obj_branch=True,
        indep_obj_scale=False,
        same_point=True,
        filter_dist=False,
        image_size=224,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        clamp=None,
        pc_sample=500,
        check_file=True,
        fhb=False,
        model_type="1encoder1decoder",
        obj_center=False
    ):
        self.input_type = input_type
        self.subsample = subsample

        self.dataset_name = dataset_name
        self.data_source = data_source
        self.image_source = image_source

        self.hand_branch = hand_branch
        self.obj_branch = obj_branch

        self.pc_sample = pc_sample

        self.filter_dist = filter_dist
        self.model_type = model_type
        self.obj_center = obj_center

        if image_source:
            self.encoder_input_source = image_source
        else:
            self.encoder_input_source = None
        (self.npyfiles_hand,
         self.npyfiles_obj,
         self.normalization_params,
         self.encoder_input_files) = data_utils.gf_get_instance_filenames(data_source, input_type, self.encoder_input_source,
                                                                          split, check_file=check_file, fhb=fhb, dataset_name=dataset_name)

        self.indep_obj_scale = indep_obj_scale
        self.same_point = same_point
        self.clamp = clamp

        logging.debug(
            "using "
            + str(len(self.npyfiles_hand))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.npyfiles_hand)

    def __getitem__(self, idx):
        filename_hand = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_hand[idx]
        )
        filename_obj = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_obj[idx]
        )

        norm_params_filename = os.path.join(
            self.data_source, misc_utils.normalization_param_subdir, self.normalization_params[
                idx]
        )

        if not self.load_ram:
            if 'image' in self.input_type:
                pass
            else:
                encoder_input_hand = data_utils.get_negative_surface_points(
                    filename_hand, self.pc_sample)
                encoder_input_obj = data_utils.get_negative_surface_points(
                    filename_obj, self.pc_sample)
            scale, offset = data_utils.unpack_normal_params(
                norm_params_filename)

            # If only hand branch or obj branch is used, subsample is not reduced by half
            # to maintain the same number of samples used when trained with two branches.
            if not self.same_point or not (self.hand_branch and self.obj_branch):
                num_sample = self.subsample
            else:
                num_sample = int(self.subsample / 2)

            hand_samples, hand_labels = data_utils.unpack_sdf_samples(
                filename_hand, num_sample, hand=True, clamp=self.clamp, filter_dist=self.filter_dist)
            obj_samples, obj_labels = data_utils.unpack_sdf_samples(
                filename_obj, num_sample, hand=False, clamp=self.clamp, filter_dist=self.filter_dist)

            if not self.indep_obj_scale:
                # Scale object back to the hand coordinate
                obj_samples[:, 0:3] = obj_samples[:, 0:3] / scale - offset
                # Scale sdf back to original scale
                obj_samples[:, 3] = obj_samples[:, 3] / scale
                hand_samples[:, 4] = hand_samples[:, 4] / scale

                # scale to fit unit sphere -> rescale when reconstruction
                obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2.0
                hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2.0

                if self.input_type == 'point_cloud':
                    encoder_input_obj = encoder_input_obj / scale - offset

            if 'VAE' in self.model_type and self.obj_center:
                # normalize point cloud
                (encoder_input_hand, encoder_input_obj,
                 hand_samples, obj_samples) = data_utils.normalize_obj_center(encoder_input_hand, encoder_input_obj,
                                                                              hand_samples, obj_samples, scale=2.0)

            return (hand_samples, hand_labels, obj_samples, obj_labels,
                    scale, offset, encoder_input_hand, encoder_input_obj, self.npyfiles_hand[idx])


class PointCloudsSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        load_ram=False,
        indep_obj_scale=False,
        pc_sample=500,
        print_filename=False,
        fhb=False,
        model_type="1encoder1decoder",
        obj_center=False,
        dataset_name='obman'
    ):
        self.data_source = data_source
        # self.imagefiles = get_images_filenames(image_source, split, fhb=fhb)

        self.pc_sample = pc_sample
        self.model_type = model_type
        self.obj_center = obj_center

        (self.npyfiles_hand,
         self.npyfiles_obj,
         self.normalization_params,
         _) = data_utils.gf_get_instance_filenames(data_source, 'point_cloud', None, split, check_file=False, fhb=fhb)

        self.indep_obj_scale = indep_obj_scale

        logging.debug(
            "using "
            + str(len(self.npyfiles_hand))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

    def __len__(self):
        return len(self.npyfiles_hand)

    def __getitem__(self, idx):
        # image_filename = os.path.join(self.image_source, self.imagefiles[idx])
        filename_hand = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_hand[idx]
        )
        filename_obj = os.path.join(
            self.data_source, misc_utils.sdf_samples_subdir, self.npyfiles_obj[idx]
        )
        norm_params_filename = os.path.join(
            self.data_source, misc_utils.normalization_param_subdir, self.normalization_params[
                idx]
        )

        if self.load_ram:
            return 0
            # return (
            #     data_utils.unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
            #     idx,
            # )
        else:
            # pc_sample = 500
            encoder_input_hand = data_utils.get_negative_surface_points(
                filename_hand, self.pc_sample)
            encoder_input_obj = data_utils.get_negative_surface_points(
                filename_obj, self.pc_sample)

            scale, offset = data_utils.unpack_normal_params(
                norm_params_filename)

            # print(obj_samples[0:2])
            if not self.indep_obj_scale:
                encoder_input_obj = encoder_input_obj / scale - offset

            if 'VAE' in self.model_type and self.obj_center:
                encoder_input_hand, encoder_input_obj = data_utils.normalize_obj_center(
                    encoder_input_hand, encoder_input_obj)
                # print("object center!!!")
            return encoder_input_hand, encoder_input_obj, idx, self.npyfiles_hand[idx]


######################################################################################
############### Older DataSets #######################################################
######################################################################################


class SingleShapeData(torch.utils.data.Dataset):
    def __init__(self, points, sdf_object) -> None:
        super().__init__()
        self.points = points
        self.sdf_object = sdf_object

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index], self.sdf_object[index]


class SingleShapeSingleGripperData(torch.utils.data.Dataset):
    def __init__(self, points, sdf_object, sdf_gripper) -> None:
        super().__init__()

        self.points = points
        self.sdf_object = sdf_object
        self.sdf_gripper = sdf_gripper

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        # return super().__getitem__(index)

        return self.points[index], self.sdf_object[index], self.sdf_gripper[index]


class SingleShapeMultiGripperData(torch.utils.data.Dataset):
    pass


class MultiShapeMultiGripperData(torch.utils.data.Dataset):
    pass
