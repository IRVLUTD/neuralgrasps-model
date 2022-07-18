import logging
import torch
import numpy as np
import warnings


################
# Triplet Loss Utils

def rearrange_triplet(triplet : tuple, cmap_dist: np.ndarray, swap=False) -> tuple:
    '''
    Rearranges the indices in the input tuple such that the first index is the appropriate
    anchor, second index is the positive example and the third index is the negative e.g.

    positive, negative tagging is based on the cmap_dist matrix 

    swap: See torch.nn.TripletMarginLoss() swap parameter.
    '''
    a, p, n = triplet
    # dist_a_i = cmap_dist[a, i]
    # dist_a_j = cmap_dist[a, j]
    # dist_i_j = cmap_dist[i, j]
    # Swap them if p is farther away than n (from a)
    if cmap_dist[a, p] > cmap_dist[a, n]:
        # Swap pos and neg
        p, n = n, p
    # Swap anchor and positive if positive is closer to neg than anchor.
    # Then the neg will be a "hard" negative for the positive (which becomes the new anchor)
    # Ref. 
    if cmap_dist[a, n] > cmap_dist[p, n]:
        a, p = p, a
    
    return (a, p, n)

def get_margins(triplets: list[tuple], cmap_dist: np.ndarray, factor=0.9) -> list[tuple]:
    '''
    Return the appropriate margin for a list of (anchor, pos, neg) samples using the 
    ground truth margin between the (anchor, pos) and (anchor, neg) using the cmap_dist

    `factor` is optional to make the learning a bit easier
    
    Assumes the triplets to be well-formed i.e d(a,n) > d(a,p), otherwise the returned
    margin will be negative
    '''
    return [(cmap_dist[a,n] - cmap_dist[a,p]) * factor for a,p,n in triplets]




############################################
######## Other Utils #######################


def decode_sdf_encoderDecoder(encoderDecoder, latent_vector, queries, gripper_pc):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf_obj, sdf_grp = encoderDecoder(gripper_pc, inputs)
    return sdf_obj, sdf_grp


def decode_multisdf(decoder, latent_vector, queries, gripper_idx, gripper_emb=None):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    elif gripper_emb is None:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        grp_emb_repeat = gripper_emb.expand(num_samples, -1)
        inputs = torch.cat([grp_emb_repeat, latent_repeat, queries], 1)

    predictions = decoder(inputs)
    return predictions[:, -1].unsqueeze(1), predictions[:, gripper_idx].unsqueeze(1)
    # return predictions[:, gripper_idx].unsqueeze(1)


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter("DeepSdf - %(levelname)s - %(message)s")
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)

    if args.logfile is not None:
        file_logger_handler = logging.FileHandler(args.logfile)
        file_logger_handler.setFormatter(formatter)
        logger.addHandler(file_logger_handler)


def threshold_min_max(tensor, min_vec, max_vec):
    return torch.min(max_vec, torch.max(tensor, min_vec))


def ce_mask_from_min_max(sdfs_hand, labels, max_vec, m_ones_vec):
    return torch.where(torch.abs(sdfs_hand) < max_vec, labels, m_ones_vec)


def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf_obj, sdf_grp = decoder(inputs)
    return sdf_obj, sdf_grp

def dsdf_decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf


def decode_sdf_class(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf, predicted_class = decoder(inputs)

    return sdf, predicted_class


def decode_sdf_multi_output(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf_hand, sdf_obj, predicted_class = decoder(inputs)

    return sdf_hand, sdf_obj, predicted_class


def customized_export_ply(outfile_name, v, f=None, v_n=None, v_c=None, f_c=None, e=None):
    # TODO: edge color
    # TODO: Face normal

    # v: Vertex position, N_v x 3 float numpy array
    # f: Face, N_f x 3 int numpy array
    # v_n: Vertex normal, N_v x 3 float numpy array
    # v_c: Vertex color, N_v x (3 or 4) uchar numpy array
    # f_n: Face normal, N_f x 3 float numpy array
    # f_c: Face color, N_f x (3 or 4) uchar numpy array
    # e: Edge, N_e x 2 int numpy array
    # mode: ascii or binary ply file. Value is {'ascii', 'binary'}

    v_n_flag = False
    v_c_flag = False
    f_c_flag = False

    N_v = v.shape[0]
    assert(v.shape[1] == 3)
    if not type(v_n) == type(None):
        assert(v_n.shape[0] == N_v)
        v_n_flag = True
    if not type(v_c) == type(None):
        assert(v_c.shape[0] == N_v)
        v_c_flag = True
        if v_c.shape[1] == 3:
            warnings.warn(
                "Vertex color does not provide alpha channel, use default alpha = 255")
            alpha_channel = np.zeros((N_v, 1), dtype=np.ubyte)+255
            v_c = np.hstack((v_c, alpha_channel))

    N_f = 0
    if not type(f) == type(None):
        N_f = f.shape[0]
        assert(f.shape[1] == 3)
        if not type(f_c) == type(None):
            assert(f_c.shape[0] == f.shape[0])
            f_c_flag = True
            if f_c.shape[1] == 3:
                warnings.warn(
                    "Face color does not provide alpha channel, use default alpha = 255")
                alpha_channel = np.zeros((N_f, 1), dtype=np.ubyte)+255
                f_c = np.hstack((f_c, alpha_channel))
    N_e = 0
    if not type(e) == type(None):
        N_e = e.shape[0]

    with open(outfile_name, 'w') as file:
        # Header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n' % (N_v))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if v_n_flag:
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
        if v_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        file.write('element face %d\n' % (N_f))
        file.write('property list uchar int vertex_indices\n')
        if f_c_flag:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('property uchar alpha\n')

        if not N_e == 0:
            file.write('element edge %d\n' % (N_e))
            file.write('property int vertex1\n')
            file.write('property int vertex2\n')

        file.write('end_header\n')

        # Main body:
        # Vertex
        if v_n_flag and v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f %d %d %d %d\n' %
                           (v[i, 0], v[i, 1], v[i, 2],
                            v_n[i, 0], v_n[i, 1], v_n[i, 2],
                               v_c[i, 0], v_c[i, 1], v_c[i, 2], v_c[i, 3]))
        elif v_n_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %f %f %f\n' %
                           (v[i, 0], v[i, 1], v[i, 2],
                            v_n[i, 0], v_n[i, 1], v_n[i, 2]))
        elif v_c_flag:
            for i in range(0, N_v):
                file.write('%f %f %f %d %d %d %d\n' %
                           (v[i, 0], v[i, 1], v[i, 2],
                            v_c[i, 0], v_c[i, 1], v_c[i, 2], v_c[i, 3]))
        else:
            for i in range(0, N_v):
                file.write('%f %f %f\n' %
                           (v[i, 0], v[i, 1], v[i, 2]))
        # Face
        if f_c_flag:
            for i in range(0, N_f):
                file.write('3 %d %d %d %d %d %d %d\n' %
                           (f[i, 0], f[i, 1], f[i, 2],
                            f_c[i, 0], f_c[i, 1], f_c[i, 2], f_c[i, 3]))
        else:
            for i in range(0, N_f):
                file.write('3 %d %d %d\n' %
                           (f[i, 0], f[i, 1], f[i, 2]))

        # Edge
        if not N_e == 0:
            for i in range(0, N_e):
                file.write('%d %d\n' % (e[i, 0], e[i, 1]))
