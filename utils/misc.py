import json
import os
import logging
import pickle

import numpy as np
import torch


data_sdf_dir = 'sdf'
data_urdf_pc = 'urdf_point_cloud'
datadir_contactmap = 'contactmap'

########################################
### Grasping Field Code ################
########################################

sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
model_params_subdir = "ModelParameters"
normalization_param_subdir = "NormalizationParameters"
optimizer_params_subdir = "OptimizerParameters"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
evaluation_subdir = "Evaluation"
specifications_filename = "specs.json"
logs_filename = "Logs.pth"

latent_codes_subdir = "LatentCodes"
data_source_map_filename = ".datasources.json"
reconstruction_codes_subdir = "Codes"

gripper_codes_subdir = "GripperCodes"

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def is_checkpoint_exist(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    return os.path.isfile(filename)


def load_model_parameters(experiment_directory, checkpoint, model):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            'model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    model.load_state_dict(data["model_state_dict"], strict=False)

    return data["epoch"]


def load_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i].cuda())

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()

def load_gripper_vectors(experiment_directory, checkpoint):
    filename = os.path.join(
        experiment_directory, gripper_codes_subdir, checkpoint + ".pth"
    )
    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a gripper code file"
            + " for checkpoint '{}'".format(experiment_directory, checkpoint)
        )
    data = torch.load(filename)
    if isinstance(data["gripper_codes"], torch.Tensor):
        num_vecs = data["gripper_codes"].size()[0]
        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["gripper_codes"][i].cuda())
        return lat_vecs
    else:

        num_embeddings, embedding_dim = data["gripper_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim)

        lat_vecs.load_state_dict(data["gripper_codes"])

        return lat_vecs.weight.data.detach()


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name + "_" + instance_name + ".ply",
    )


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir

def get_gripper_codes_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, gripper_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        # instance_name + ".npz",
        "obj.npz",
    )



