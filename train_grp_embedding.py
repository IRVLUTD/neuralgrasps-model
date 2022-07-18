import re
import signal
import sys
import os
import logging
import math
import json
import time
import itertools
import random

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import utils.dataset
import utils.misc as ws
import utils.train_utils
import models.networks as arch


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return "ConstantLearningRateSchedule"

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def __repr__(self) -> str:
        return "StepLearningRateSchedule"

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def __repr__(self) -> str:
        return "WarmupLearningRateSchedule"

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(
                schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(
        experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


def save_gripper_codes(experiment_directory, filename, codes, epoch):
    gripper_codes_dir = ws.get_gripper_codes_dir(experiment_directory, True)

    all_latents = codes.state_dict()

    torch.save(
        {"epoch": epoch, "gripper_codes": all_latents},
        os.path.join(gripper_codes_dir, filename),
    )

# TODO: duplicated in workspace


def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]

def load_gripper_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_gripper_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'gripper code state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["gripper_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["gripper_codes"].size()[0]:
            raise Exception(
                "num gripper codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["gripper_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["gripper_codes"].size()[2]:
            raise Exception("gripper code dimensionality mismatch")

        for i, lat_vec in enumerate(data["gripper_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["gripper_codes"])

    return data["epoch"]

def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model, writer, step):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())
        writer.add_scalar('Misc/ParamMag_' + name + 'mag',
                          param.data.norm().item(), step)


def main_function(experiment_directory, continue_from, batch_split):

    tensorboard_logdir = os.path.join(experiment_directory, 'logs')
    if not os.path.isdir(tensorboard_logdir):
        os.mkdir(tensorboard_logdir)

    writer = SummaryWriter(log_dir=tensorboard_logdir)

    logging.debug("running " + experiment_directory)

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    # arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.debug(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)

    # NEW CODE!
    gripper_loss_weight = get_spec_with_default(specs, "GripperWeight", 0.5)

    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory,
                       "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory,
                            "latest.pth", lat_vecs, epoch)
        save_gripper_codes(experiment_directory,
                           "latest.pth", grp_embeddings, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory,
                   str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(
            epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(
            epoch) + ".pth", lat_vecs, epoch)
        save_gripper_codes(experiment_directory,
                           str(epoch) + ".pth",  grp_embeddings, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(
        specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(
        specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    # Flags to enforce similarity amongst the latent vectors
    enforce_code_similarity = get_spec_with_default(
        specs, "EnforceCodeSimilarity", False)
    variable_triplet_margin = get_spec_with_default(
        specs, "TripletVariableMargin", False)

    grp_embedding_size = get_spec_with_default(
        specs, "GripperEmbeddingLength", 50)

    grp_code_bound = get_spec_with_default(specs, "GrpCodeBound", None)
    grp_do_code_regularization = get_spec_with_default(
        specs, "GripperCodeRegularization", True)
    grp_code_lambda = get_spec_with_default(
        specs, "GrpCodeRegLambda", 1e-4)

    decoder = arch.dsdfDecoder(
        grp_embedding_size + latent_size,
        **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    split_fpath = os.path.join(experiment_directory, train_split_file)
    if not os.path.isfile(split_fpath):
        print(
            f"Split file should be present in experiment directory: {experiment_directory}")
        print("Exiting ...")
        exit(0)
    with open(split_fpath, "r") as f:
        train_split = json.load(f)

    # sdf_dataset = utils.dataset.SDFSamples(
    #     data_source, train_split, num_samp_per_scene)
    sdf_dataset = utils.dataset.MultiGripperSamples(
        data_source, train_split, num_samp_per_scene)

    num_data_loader_threads = get_spec_with_default(
        specs, "DataLoaderThreads", 4)
    logging.debug("loading data with {} threads".format(
        num_data_loader_threads))

    sdf_loader = torch.utils.data.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.debug(decoder)

    grp_embeddings = torch.nn.Embedding(
        sdf_dataset.num_grippers, grp_embedding_size, max_norm=grp_code_bound)
    torch.nn.init.normal_(
        grp_embeddings.weight.data,
        0.0,
        1.0/math.sqrt(grp_embedding_size)
    )

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev",
                              1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    logging.debug(
        "initialized Gripper Embeddingswith mean magnitude {}".format(
            get_mean_latent_vector_magnitude(grp_embeddings)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    # Triplet Loss for enforcing a metric over the latent embedding space
    triplet_loss = torch.nn.TripletMarginLoss(reduction="mean")
    fixed_margin = 3.0
    triplet_loss_custom_margin = torch.nn.TripletMarginLoss(
        margin=fixed_margin, reduction="none")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": grp_embeddings.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            }
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        grpemb_epoch = load_gripper_vectors(
            experiment_directory, continue_from + ".pth", grp_embeddings
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch and model_epoch == grpemb_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    logging.info(
        "Number of gripper embedding parameters: {} (# codes {}, code dim {})".format(
            grp_embeddings.num_embeddings * grp_embeddings.embedding_dim,
            grp_embeddings.num_embeddings,
            grp_embeddings.embedding_dim,
        )
    )

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        epoch_loss_overall = 0
        epoch_loss_obj = 0
        epoch_loss_grp = 0
        epoch_loss_l2size = 0
        epoch_loss_reg = 0
        epoch_loss_reg_grp = 0
        epoch_loss_triplet = 0

        for idx_dataloader, (grp_idxs, sdf_data, indices, _) in enumerate(sdf_loader):

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 5)
            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False

            xyz = sdf_data[:, :3]
            sdf_gt_obj = sdf_data[:, 3].unsqueeze(1)
            sdf_gt_grp = sdf_data[:, 4].unsqueeze(1)

            if enforce_minmax:
                sdf_gt_obj = torch.clamp(sdf_gt_obj, minT, maxT)
                sdf_gt_grp = torch.clamp(sdf_gt_grp, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)

            # Save a copy for the contactmap constraint
            org_indices = list(indices)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            grp_idxs = torch.chunk(
                grp_idxs.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt_obj = torch.chunk(sdf_gt_obj, batch_split)
            sdf_gt_grp = torch.chunk(sdf_gt_grp, batch_split)

            optimizer_all.zero_grad()

            batch_loss = 0.0

            latent_code_metric_loss = 0.0
            # Have a loss over the Z space vectors for similar grasps.
            if enforce_code_similarity:
                total_triplets = len(org_indices)
                triplets = [tuple(random.sample(org_indices, k=3))
                            for _ in range(total_triplets)]
                triplets = [utils.train_utils.rearrange_triplet(
                    t, sdf_dataset.cmap_dist, swap=True) for t in triplets]

                anchor_idxs = torch.LongTensor([t[0] for t in triplets])
                posvec_idxs = torch.LongTensor([t[1] for t in triplets])
                negvec_idxs = torch.LongTensor([t[2] for t in triplets])
                anchors = lat_vecs(anchor_idxs)
                pos_vec = lat_vecs(posvec_idxs)
                neg_vec = lat_vecs(negvec_idxs)

                if not variable_triplet_margin:
                    latent_code_metric_loss = triplet_loss(
                        anchors, pos_vec, neg_vec)
                else:
                    tmp_triplet_loss = triplet_loss_custom_margin(
                        anchors, pos_vec, neg_vec)
                    margins = torch.tensor(utils.train_utils.get_margins(
                        triplets, sdf_dataset.cmap_dist), dtype=float)
                    # tmp_triplet_loss = 2.0 - D(a, n, p) where
                    # D(a, n, p) = d(a, n) - d(a, p)
                    tmp_triplet_loss = -1 * tmp_triplet_loss + fixed_margin
                    # This just gives us D(a, n, p)
                    margin_mask = tmp_triplet_loss < margins
                    batch_triplet_losses = margins[margin_mask] - \
                        tmp_triplet_loss[margin_mask]
                    latent_code_metric_loss = torch.mean(batch_triplet_losses)
            epoch_loss_triplet += latent_code_metric_loss.item()

            latent_code_metric_loss.backward()

            for i in range(batch_split):
                batch_gripper_vecs = grp_embeddings(grp_idxs[i])
                batch_vecs = lat_vecs(indices[i])
                input_dsdf = torch.cat(
                    [batch_gripper_vecs, batch_vecs, xyz[i]], dim=1).cuda()

                # NN optimization
                pred_sdf_obj, pred_sdf_grp = decoder(input_dsdf)

                if enforce_minmax:
                    pred_sdf_obj = torch.clamp(pred_sdf_obj, minT, maxT)
                    pred_sdf_grp = torch.clamp(pred_sdf_grp, minT, maxT)

                chunk_loss_obj = loss_l1(
                    pred_sdf_obj, sdf_gt_obj[i].cuda()) / num_sdf_samples
                chunk_loss_grp = loss_l1(
                    pred_sdf_grp, sdf_gt_grp[i].cuda()) / num_sdf_samples

                chunk_loss = (1 - gripper_loss_weight) * chunk_loss_obj + \
                    gripper_loss_weight * chunk_loss_grp

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + reg_loss.cuda()
                    epoch_loss_l2size += l2_size_loss.item()
                    epoch_loss_reg += reg_loss

                if grp_do_code_regularization:
                    grp_l2_loss = torch.sum(
                        torch.norm(batch_gripper_vecs, dim=1))
                    grp_reg_loss = (
                        grp_code_lambda * min(1, epoch / 100) * grp_l2_loss
                    ) / num_sdf_samples

                    chunk_loss = chunk_loss + grp_reg_loss.cuda()
                    epoch_loss_reg_grp += grp_reg_loss

                chunk_loss.backward()

                batch_loss += chunk_loss.item()
                epoch_loss_overall += batch_loss
                epoch_loss_obj += chunk_loss_obj.item()
                epoch_loss_grp += chunk_loss_grp.item()

            logging.debug("loss = {}".format(batch_loss))
            loss_log.append(batch_loss)

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    decoder.parameters(), grad_clip)

            optimizer_all.step()

        # Tensorboard -- collect the losses
        writer.add_scalar('Loss/overall', epoch_loss_overall, epoch)
        writer.add_scalar('Loss/object', epoch_loss_obj, epoch)
        writer.add_scalar('Loss/gripper', epoch_loss_grp, epoch)
        writer.add_scalar('Loss/l2_size', epoch_loss_l2size, epoch)
        writer.add_scalar('Loss/reg_loss', epoch_loss_reg, epoch)
        writer.add_scalar('Loss/reg_grp', epoch_loss_reg_grp, epoch)
        if enforce_code_similarity:
            writer.add_scalar('Loss/SimLoss', epoch_loss_triplet, epoch)

        end = time.time()
        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)
        writer.add_scalar('Misc/time-per-epoch', seconds_elapsed, epoch)

        lr_log.append([schedule.get_learning_rate(epoch)
                      for schedule in lr_schedules])
        for i, schedule in enumerate(lr_schedules):
            writer.add_scalar(f"LR/{str(schedule)}-{i}",
                              schedule.get_learning_rate(epoch), epoch)

        mean_lv_mag = get_mean_latent_vector_magnitude(lat_vecs).item()
        lat_mag_log.append(mean_lv_mag)
        writer.add_scalar('Misc/mean-lv-mag', mean_lv_mag, epoch)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )

            # Tensorboard -- Log the embeddings
            # writer.add_embedding(np.array(lat_vecs.weight.data), metadata=list(
            #     range(len(sdf_dataset))), global_step=epoch)

        writer.flush()

    writer.close()


if __name__ == "__main__":
    # Example run: python train_deep_sdf.py -e experiments/epochs_2000_weight_0.5
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Train a DeepSDF autodecoder")
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
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    utils.train_utils.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    utils.train_utils.configure_logging(args)

    main_function(args.experiment_directory,
                  args.continue_from, int(args.batch_split))
