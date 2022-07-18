#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
# from turtle import end_fill
import torch

# import deep_sdf
import utils.misc as ws
import utils.data_utils
import utils.train_utils
import utils.mesh
import models.networks as arch


def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
    gripper_weight=0.5
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        lr = initial_lr * ((1 / decreased_by) **
                           (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = utils.data_utils.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()
        xyz = sdf_data[:, 0:3]
        sdf_gt_obj = sdf_data[:, 3].unsqueeze(1)
        sdf_gt_grp = sdf_data[:, 4].unsqueeze(1)

        sdf_gt_obj = torch.clamp(sdf_gt_obj, -clamp_dist, clamp_dist)
        sdf_gt_grp = torch.clamp(sdf_gt_grp, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf_obj, pred_sdf_grp = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf_obj, pred_sdf_grp = decoder(inputs)

        pred_sdf_obj = torch.clamp(pred_sdf_obj, -clamp_dist, clamp_dist)
        pred_sdf_grp = torch.clamp(pred_sdf_grp, -clamp_dist, clamp_dist)

        loss = (1 - gripper_weight) * loss_l1(pred_sdf_obj, sdf_gt_obj) + \
            gripper_weight * loss_l1(pred_sdf_grp, sdf_gt_grp)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


if __name__ == "__main__":

    # Sample Run:
    # python reconstruct_latent.py -e experiments/ -d /home/ninad/Desktop/Docs/phd-res/proj-irvl-grasp-transfer/code/docker-data/output_dataset/ -s split_validation.json --skip

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    utils.train_utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    utils.train_utils.configure_logging(args)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    with open(specs_filename, "r") as f:
        specs = json.load(f)

    # arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]
    gripper_weight = specs["GripperWeight"]

    decoder = arch.dsdfDecoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    split_f = os.path.join(args.experiment_directory, args.split_filename)
    with open(split_f, "r") as f:
        split = json.load(f)

    # npz_filenames = utils.data_utils.dsdf_get_instance_filenames(
    #     args.data_source, split)
    cmap_f, grp_names, gpc_f, npz_filenames = utils.data_utils.get_instance_filelist(
        args.data_source, split)

    random.shuffle(npz_filenames)

    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = True
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(
            saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    for ii, npz in enumerate(npz_filenames):

        if "npz" not in npz:
            continue

        # full_filename = os.path.join(
        #     args.data_source, ws.sdf_samples_subdir, npz)

        full_filename = npz
        logging.debug("loading {}".format(npz))

        data_sdf = utils.data_utils.read_sdf_samples_into_ram(
            full_filename)

        filename_split_list = os.path.normpath(
            full_filename).split(os.path.sep)
        npz_file_info = '-'.join(filename_split_list[-4:])[:-4]
        # EXAMPLE:
        # filename_split_list[-4:] Gives Us:
        # ['003_cracker_box_google_16k_textured_scale_1000',
        # 'sdf',
        # 'fetch_gripper',
        # 'sdf_graspnum_40.npz']
        # '-'.join(filename_split_list[-4:]) Gives us the name with .npz at end
        # So having a final [:-4] removes the ".npz" part

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz_file_info +
                    "-" + str(k + rerun)
                )
                mesh_filename_object = os.path.join(
                    reconstruction_meshes_dir, npz_file_info +
                    "-" + str(k + rerun) + '_obj'
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz_file_info +
                    "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, npz_file_info)
                mesh_filename_object = os.path.join(
                    reconstruction_meshes_dir, npz_file_info + '_obj')
                latent_filename = os.path.join(
                    reconstruction_codes_dir, npz_file_info + ".pth"
                )

            if (
                args.skip
                # and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                0.01,  # [emp_mean,emp_var],
                0.1,
                num_samples=8000,
                lr=5e-3,
                l2reg=True,
                gripper_weight=gripper_weight
            )
            logging.debug("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()

            if not os.path.exists(os.path.dirname(mesh_filename)):
                os.makedirs(os.path.dirname(mesh_filename))

            # if not save_latvec_only:
            #     start = time.time()
            #     with torch.no_grad():
            #         # Mesh for the gripper
            #         utils.mesh.create_mesh_custom(
            #             decoder, latent, mesh_filename, N=256, max_batch=int(2 ** 18), isGripper=True
            #         )
            #         # Mesh for the object
            #         utils.mesh.create_mesh_custom(
            #             decoder, latent, mesh_filename_object, N=256, max_batch=int(2 ** 18), isGripper=False
            #         )
            #     logging.debug("total time: {}".format(time.time() - start))

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
