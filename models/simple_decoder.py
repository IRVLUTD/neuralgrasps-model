import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleShapeDecoder(nn.Module):

    def __init__(self, inner_dim=512, num_layers=8, dropout_rate=0.3, output_dim=2):
        super(SingleShapeDecoder, self).__init__()

        self.layers = nn.ModuleList()

        input_dim = 3
        # output_dim = 2
        self.layers.append(nn.Linear(input_dim, inner_dim))
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(inner_dim, inner_dim))
        # output 2 sdf values, one to object, one to gripper
        self.output = nn.Linear(inner_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()

    # input is N x 3 i.e no latent vector
    def forward(self, input):
        x = input
        for layer in self.layers:
            x = F.relu(layer(x), inplace=True)
            x = self.dropout(x)

        x = self.tanh(self.output(x))
        # x will be Nx2
        return x


class AD_SDF(nn.Module):
    def __init__(self, z_dim=256, data_shape=200):
        super(AD_SDF, self).__init__()
        self.decoder_stage1 = nn.Sequential(
            nn.Linear(z_dim+3, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 253))

        self.decoder_stage2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 2),
            nn.Tanh())

        self.latent_vectors = nn.Parameter(
            torch.FloatTensor(data_shape, z_dim))

        init.xavier_normal(self.latent_vectors)

    def forward(self, ind, x):
        code = self.latent_vectors[ind].repeat(x.shape[0], 1)
        data = torch.cat((code, x), dim=1)
        decoder_stage1_out = self.decoder_stage1(data)
        data = torch.cat((decoder_stage1_out, data), dim=1)
        decoder_stage2_out = self.decoder_stage2(data)
        return decoder_stage2_out

    def codes(self):
        return self.latent_vectors
