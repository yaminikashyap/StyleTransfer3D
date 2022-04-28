import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm
from styleatlasnet import StyleAtlasnet

# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# class PointNet(torch.nn.Module):
#     def __init__(self, num_input_features, num_output_features):
#         super(PointNet, self).__init__()
#         self.mlp1 = MLP([num_input_features, 32, 64, 128])
#         self.mlp2 = MLP([128, 128])
#         self.mlp3 = MLP([256, 128, 64])
#         self.fc1 = torch.nn.Linear(64, num_output_features, bias=True)

#     def forward(self, x):
#         f_i = self.mlp1(x)
#         h_i = self.mlp2(f_i)
#         pool = torch.max(h_i, 0)
#         g = pool.values.repeat((int)(x.shape[0]), 1)
#         concat = torch.cat((g, f_i), 1)
#         t = self.mlp3(concat)
#         y_i = self.fc1(t)
#         return y_i
class PointNet(nn.Module):
    def __init__(self, nlatent=1024, dim_input=3, normalization='bn', activation='relu'):
        """
        PointNet Encoder
        See : PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
                Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
        """

        super(PointNet, self).__init__()
        self.dim_input = dim_input
        if normalization == 'sn':
            self.conv1 = SpectralNorm(torch.nn.Conv1d(dim_input, 64, 1))
            self.conv2 = SpectralNorm(torch.nn.Conv1d(64, 128, 1))
            self.conv3 = SpectralNorm(torch.nn.Conv1d(128, nlatent, 1))
            self.lin1 = SpectralNorm(nn.Linear(nlatent, nlatent))
            self.lin2 = SpectralNorm(nn.Linear(nlatent, nlatent))
        else:
            self.conv1 = torch.nn.Conv1d(dim_input, 64, 1)
            self.conv2 = torch.nn.Conv1d(64, 128, 1)
            self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
            self.lin1 = nn.Linear(nlatent, nlatent)
            self.lin2 = nn.Linear(nlatent, nlatent)

        norm = torch.nn.BatchNorm1d if normalization == 'bn' else nn.Identity
        self.bn1 = norm(64)
        self.bn2 = norm(128)
        self.bn3 = norm(nlatent)
        self.bn4 = norm(nlatent)
        self.bn5 = norm(nlatent)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.nlatent = nlatent

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.nlatent)
        x = self.activation(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = self.activation(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)

class ThreeDsnet(nn.Module):
    def __init__(self):
        super(ThreeDsnet, self).__init__()

        self.content_encoder_0 = self.content_encoder_1 = PointNet(1024)
        
        self.style_encoder = PointNet(512)

        self.decoder_0 = self.decoder_1 = StyleAtlasnet(2500,25)

        self.discriminator_encoder= PointNet(1024)
        self.discriminator_0 = nn.Sequential(self.discriminator_encoder, MLP([1024,1]))
        self.discriminator_1 = nn.Sequential(self.discriminator_encoder, MLP([1024,1]))
    

    def forward(self, data0, data1, train=True):
        # Encode
        content_0 = self.content_encoder_0(data0)
        content_1 = self.content_encoder_1(data1)

        style_0_prime = self.style_encoder(data0)
        style_1_prime = self.style_encoder(data1)

        # Decode latent codes (within domain)
        out_00 = (self.decoder_0(content_0, style_0_prime, train=train)['points_3']).transpose(2, 3).contiguous()
        out_00 = out_00.view(out_00.size(0), -1, 3)
        out_11 = (self.decoder_0(content_1, style_1_prime, train=train)['points_3']).transpose(2, 3).contiguous()
        out_11 = out_11.view(out_11.size(0), -1, 3)
        # Decode latent codes (cross domain)
        out_01 = (self.decoder_0(content_0, style_1_prime, train=train)['points_3']).transpose(2, 3).contiguous()
        out_01 = out_01.view(out_01.size(0), -1, 3)
        out_10 = (self.decoder_0(content_1, style_0_prime, train=train)['points_3']).transpose(2, 3).contiguous()
        out_10 = out_10.view(out_10.size(0), -1, 3)


        content_01 = self.content_encoder_1(out_01.transpose(1,2))
        style_01 = self.style_encoder(out_01.transpose(1,2))
        content_10 = self.content_encoder_0(out_10.transpose(1,2))
        style_10 = self.style_encoder(out_10.transpose(1,2))

        cycle_out_010 = self.decoder_0(content_01, style_0_prime, train=train)
        cycle_out_101 = self.decoder_1(content_10, style_1_prime, train=train)

        # Classify domain membership for each style transferred pointcloud
        class_00 = self.discriminator_0(out_00.transpose(1,2))
        class_01 = self.discriminator_1(out_01.transpose(1,2))
        class_11 = self.discriminator_1(out_11.transpose(1,2))
        class_10 = self.discriminator_0(out_10.transpose(1,2))

        class_0 = self.discriminator_0(data0)
        class_1 = self.discriminator_1(data1)

        return {"reconstructed_outputs":[out_00, out_11, out_01, out_10],
        "content_encoder_outputs":[content_01, content_10],
        "content_encoder_prime":[content_0, content_1],
        "style_encoder_primes":[style_0_prime, style_1_prime],
        "style_encoder_reconstructed_outputs":[style_01, style_10],
        "cycle_reconstructed_outputs":[cycle_out_010, cycle_out_101],
        "discriminator_outputs":[class_00, class_01, class_10, class_11, class_0, class_1]}
