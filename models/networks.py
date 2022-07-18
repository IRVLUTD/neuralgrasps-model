import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torchvision import datasets, models, transforms



# From the original deepSDF codebase
# Expects a specs.json file like this:
#  "NetworkSpecs" : {
#     "dims" : [ 512, 512, 512, 512, 512, 512, 512, 512 ],
#     "dropout" : [0, 1, 2, 3, 4, 5, 6, 7],
#     "dropout_prob" : 0.2,
#     "norm_layers" : [0, 1, 2, 3, 4, 5, 6, 7],
#     "latent_in" : [4],
#     "xyz_in_all" : false,
#     "use_tanh" : false,
#     "latent_dropout" : false,
#     "weight_norm" : true
#     },
class dsdfDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(dsdfDecoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [2]  # should be 2 for our case.

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                # so that it still remains 512 length
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer),
                        nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            # BUT WHY is this here???
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob,
                                  training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        # x will be Nx2
        return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)

class MultiGripperDecoder(nn.Module):
    def __init__(self, gripper_encoder, sdf_decoder, num_samp_per_scene):
        super(MultiGripperDecoder, self).__init__()
        self.encoder = gripper_encoder
        self.decoder = sdf_decoder
        self.num_samp_per_scene = num_samp_per_scene

    def forward(self, pc_input, queries):
        # queries should be whatever was passed to the the org dSDF auto-decoder
        pc_latent = self.encoder(pc_input)        
        pc_encoding = pc_latent.repeat_interleave(self.num_samp_per_scene, dim=0)
        decoder_inputs = torch.cat([pc_encoding, queries], 1)
        sdf_obj, sdf_gripper = self.decoder(decoder_inputs)
        return sdf_obj, sdf_gripper

# Multi-SDF Decoder
class MultiSDFDecoder(nn.Module):

    def __init__(
        self,
        latent_size,
        num_sdf_out,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(MultiSDFDecoder, self).__init__()

        def make_sequence():
            return []

        # num_sdf_out = K+1 where K is the number of grippers and +1 for the object sdf.
        dims = [latent_size + 3] + dims + [num_sdf_out]
        self.num_sdf_out = num_sdf_out
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                # so that it still remains 512 length
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer),
                        nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (D+L+3)
    ## L = latent code size
    ## D = Gripper embedding size
    ## 3 = query point
    # Output: N x (K+1) -- K sdfs for K grippers and 1 for object sdf 
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            # BUT WHY is this here???
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob,
                                  training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        # x will be N x num_sdfs
        # Return a list of SDF predictions
        # x.shape[1] == self.num_sdfs
        # return [x[:, i].unsqueeze(1) for i in range(x.shape[1])]
        return x # Better to return a N x (K+1) matrix


